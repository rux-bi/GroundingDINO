import os
import torch
import torch.nn as nn
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
from inference_on_a_image import load_image, plot_boxes_to_image
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import onnx
import onnxruntime as ort
import numpy as np
from torch.utils.benchmark import Timer, Measurement

class ModelWrapper(nn.Module):
    def __init__(self, original_model, box_threshold=0.3, text_threshold=0.25):
        super().__init__()
        self.model = original_model
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
    
    def forward(self, samples, 
                input_ids , 
                attention_mask,
                position_ids,
                token_type_ids,
                text_self_attention_masks):
        model_outputs = self.model(samples, input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks)
        logits = model_outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = model_outputs["pred_boxes"][0]  # (nq, 4)
        
        # Partition based on confidence
        max_conf = logits.max(dim=1)[0]
        conf_mask = max_conf > self.box_threshold
        num_confident = conf_mask.sum()
        # Sort based on confidence
        conf_indices = torch.argsort(conf_mask.to(torch.float32), descending=True)
        logits = logits[conf_indices]  # Rearrange logits
        boxes = boxes[conf_indices]    # Rearrange boxes
        return logits, boxes, num_confident
    
    def postprocess(self, logits, boxes, num_dets, input_ids_raw):
        pred_phrases = []
        
        for logit in logits[:num_dets]:
            posmap = logit > self.text_threshold
            posmap[0] = False
            posmap[-1] = False
            non_zero_idx = posmap.nonzero(as_tuple=True)[0]
            pred_phrase =  self.model.tokenizer.decode(input_ids_raw[non_zero_idx])
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        return boxes[:num_dets], pred_phrases


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    
    #modified config
    args.use_checkpoint = False
    args.use_transformer_ckpt = False
    
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu") 
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model


def tokenize_text_prompt(tokenizer, caption, max_text_len, device):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    captions = [caption]
    # encoder texts
    tokenized = tokenizer(captions, padding="longest", return_tensors="pt").to(device)
    specical_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    
    (
        text_self_attention_masks,
        position_ids,
        cate_to_token_mask_list,
    ) = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, specical_tokens, tokenizer)

    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[
            :, : max_text_len, : max_text_len]
        
        position_ids = position_ids[:, : max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, : max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, : max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : max_text_len]
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"].to(bool),
        "token_type_ids": tokenized["token_type_ids"],
        "position_ids": position_ids,
        "text_self_attention_masks": text_self_attention_masks
    }


def export_onnx(model_cpu, model, image, caption, export_onnx_model, box_threshold=0.3, text_threshold=0.25):

    device = "cuda"
    model = model.to(device)
    image = image.to(device)    # encoder texts
    tokenized = tokenize_text_prompt(model.tokenizer, caption, model.max_text_len, device)
    dynamic_axes={
       "input_ids": {0: "batch_size", 1: "seq_len"},
       "attention_mask": {0: "batch_size", 1: "seq_len"},
       "position_ids": {0: "batch_size", 1: "seq_len"},
       "token_type_ids": {0: "batch_size", 1: "seq_len"},
       "text_token_mask": {0: "batch_size", 1: "seq_len", 2: "seq_len"},       
       "img": {0: "batch_size"},
       "logits": {0: "batch_size"},
       "boxes": {0: "batch_size"},
    }
    #export onnx model
    onnx_path = "/offboard/GroundingDINO/.asset/grounded.onnx"
    model_wrapper = ModelWrapper(model_cpu, box_threshold, text_threshold)
    input_ids_cpu = tokenized["input_ids"].cpu()
    if export_onnx_model:
        with torch.no_grad():
            image_cpu = image[None].cpu()
            attention_mask_cpu = tokenized["attention_mask"].cpu().to(bool)
            position_ids_cpu = tokenized["position_ids"].cpu()
            token_type_ids_cpu = tokenized["token_type_ids"].cpu()
            text_self_attention_masks_cpu = tokenized["text_self_attention_masks"].cpu()
            
            logits, boxes, _ = model_wrapper(image_cpu, input_ids_cpu, attention_mask_cpu, position_ids_cpu, token_type_ids_cpu, text_self_attention_masks_cpu)
            torch.onnx.export(
                model_wrapper,
                f=onnx_path,
                args=(image_cpu, input_ids_cpu, attention_mask_cpu, position_ids_cpu, token_type_ids_cpu, text_self_attention_masks_cpu),
                input_names=["img" , "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"],
                output_names=["logits", "boxes", "num_dets"],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                export_params=True, 
                opset_version=20)
        ######################### onnx shape inference
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)
        print('Model was successfully converted to ONNX format.')
    
    trt_cache_path = "/offboard/GroundingDINO/.asset/trt_cache"
    providers = [
        ('TensorrtExecutionProvider', {
            # 'device_id': 0,                       # Select GPU to execute
            "trt_engine_cache_enable": True,
            'trt_engine_cache_path': trt_cache_path,
            'trt_fp16_enable': True,              # Enable FP16 precision for faster inference  
            # 'trt_layer_norm_fp32_fallback': True, 
        }),
    ]
    sess_opt = ort.SessionOptions()
    # sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_opt.log_severity_level = 0
    sess = ort.InferenceSession(onnx_path, providers=providers, sess_options=sess_opt)
    io_binding = sess.io_binding()
    device_type=image.device.type
    image = image[None]
    binded_logits = torch.zeros((model.num_queries, model.max_text_len), dtype=torch.float32, device='cuda')
    binded_boxes  = torch.zeros((model.num_queries, 4), dtype=torch.float32, device='cuda')
    binded_num_dets = torch.zeros((1,), dtype=torch.int64, device='cuda')
    io_binding.bind_input(name='img', device_type=device_type, device_id=0, element_type=np.float32, shape=image.shape, buffer_ptr=image.data_ptr())
    io_binding.bind_input(name='input_ids', device_type=device_type, device_id=0, element_type=np.int64, shape=tokenized["input_ids"].shape, buffer_ptr=tokenized["input_ids"].data_ptr())
    io_binding.bind_input(name='attention_mask', device_type=device_type, device_id=0, element_type=bool, shape=tokenized["attention_mask"].shape, buffer_ptr=tokenized["attention_mask"].data_ptr())
    io_binding.bind_input(name='position_ids', device_type=device_type, device_id=0, element_type=np.int64, shape=tokenized["position_ids"].shape, buffer_ptr=tokenized["position_ids"].data_ptr())
    io_binding.bind_input(name='token_type_ids', device_type=device_type, device_id=0, element_type=np.int64, shape=tokenized["token_type_ids"].shape, buffer_ptr=tokenized["token_type_ids"].data_ptr())
    io_binding.bind_input(name='text_token_mask', device_type=device_type, device_id=0, element_type=bool, shape=tokenized["text_self_attention_masks"].shape, buffer_ptr=tokenized["text_self_attention_masks"].data_ptr())
    
    io_binding.bind_output(name='logits', device_type=device_type, device_id=0, element_type=np.float32, shape=binded_logits.shape, buffer_ptr=binded_logits.data_ptr())
    io_binding.bind_output(name='boxes', device_type=device_type, device_id=0, element_type=np.float32, shape=binded_boxes.shape, buffer_ptr=binded_boxes.data_ptr())
    io_binding.bind_output(name='num_dets', device_type=device_type, device_id=0, element_type=np.int64, shape=(), buffer_ptr=binded_num_dets.data_ptr())
    sess.run_with_iobinding(io_binding)
    with torch.no_grad():
        timer_pytorch = Timer(
            # The computation which will be run in a loop and timed.
            stmt="model(image, input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks)",
            setup="""
            """,
            
            globals={
                "image": image,
                "model": model,
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "position_ids": tokenized["position_ids"],
                "token_type_ids": tokenized["token_type_ids"],
                "text_self_attention_masks": tokenized["text_self_attention_masks"],
            },
            # Control the number of threads that PyTorch uses. (Default: 1)
            num_threads=1,
        )
        timer_onnx = Timer(
            # The computation which will be run in a loop and timed.
            stmt="sess.run_with_iobinding(io_binding)",
            # `setup` will be run before calling the measurement loop, and is used to
            # populate any state which is needed by `stmt`
            setup="""
            """,
            
            globals={
                "sess": sess,
                "io_binding": io_binding,
            },
            # Control the number of threads that PyTorch uses. (Default: 1)
            num_threads=1,
        )
        m_torch: Measurement = timer_pytorch.blocked_autorange(min_run_time=1)
        m_onnx: Measurement = timer_onnx.blocked_autorange(min_run_time=1)
    print(m_torch)
    print(m_onnx)
    return model_wrapper.postprocess(binded_logits, binded_boxes, binded_num_dets, tokenized["input_ids"][0])


def ensure_checkpoint_exists(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Downloading...")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        os.system('wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O ' + checkpoint_path)
        if os.path.exists(checkpoint_path):
            print("Checkpoint downloaded successfully!")
        else:
            raise RuntimeError("Failed to download checkpoint")


if __name__ == "__main__":
    # cfg
    config_file = "/offboard/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
    checkpoint_path = "/offboard/GroundingDINO/weights/groundingdino_swint_ogc.pth"  # change the path of the model
    output_dir = "/offboard/GroundingDINO/logs"
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # ensure checkpoint exists
    ensure_checkpoint_exists(checkpoint_path)
    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=True)
    model_cpu = load_model(config_file, checkpoint_path, cpu_only=True)
    output_dir = "/offboard/GroundingDINO/logs"
    image_path = "/offboard/GroundingDINO/.asset/zed2.png"
    text_prompt = "bicycle. skateboard. door handle. luggage case. container. shoe box"
    
    image_pil, image = load_image(image_path)
    #export onnx and tensorrt engine
    export_onnx_model = True
    boxes_filt, pred_phrases = export_onnx(model_cpu, model, image, text_prompt, export_onnx_model)
    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt.cpu(),
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    image_with_box.save(os.path.join(output_dir, "pred.jpg"))
    



    
