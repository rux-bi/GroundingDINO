import argparse
import os
import torch

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
from inference_on_a_image import load_image, plot_boxes_to_image
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
# from tools.symbolic_shape_infer import SymbolicShapeInference
import onnx
# from openvino.tools.mo import convert_model
# from openvino.runtime import serialize

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


def export_openvino(model_cpu, model, image, caption, output_dir, with_logits=True, box_threshold=0.3, text_threshold=0.25, token_spans=None):
    # caption =  "the running dog ." #". ".join(input_text)
    # input_ids =  model.tokenizer([caption], return_tensors="pt")["input_ids"]
    # position_ids = torch.tensor([[0, 0, 1, 2, 3, 0]])
    # token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0]])
    # attention_mask = torch.tensor([[True, True, True, True, True, True]])
    # text_token_mask = torch.tensor([[[ True, False, False, False, False, False],
    #      [False,  True,  True,  True,  True, False],
    #      [False,  True,  True,  True,  True, False],
    #      [False,  True,  True,  True,  True, False],
    #      [False,  True,  True,  True,  True, False],
    #      [False, False, False, False, False,  True]]])
    
    # img = torch.randn(1, 3, 800, 1200)
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda"
    model = model.to(device)
    image = image.to(device)
    model_cpu.eval()
    captions = [caption]
    # encoder texts
    tokenized = model.tokenizer(captions, padding="longest", return_tensors="pt").to(device)
    specical_tokens = model.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    
    (
        text_self_attention_masks,
        position_ids,
        cate_to_token_mask_list,
    ) = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, specical_tokens, model.tokenizer)

    if text_self_attention_masks.shape[1] > model.max_text_len:
        text_self_attention_masks = text_self_attention_masks[
            :, : model.max_text_len, : model.max_text_len]
        
        position_ids = position_ids[:, : model.max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, : model.max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, : model.max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : model.max_text_len]
    dynamic_axes={
       "input_ids": {0: "batch_size", 1: "seq_len"},
       "attention_mask": {0: "batch_size", 1: "seq_len"},
       "position_ids": {0: "batch_size", 1: "seq_len"},
       "token_type_ids": {0: "batch_size", 1: "seq_len"},
       "text_token_mask": {0: "batch_size", 1: "seq_len", 2: "seq_len"},       
       "img": {0: "batch_size", 2: "height", 3: "width"},
       "logits": {0: "batch_size"},
       "boxes": {0: "batch_size"}
    }
    # import pdb;pdb.set_trace()
    #export onnx model
    onnx_path = "/offboard/GroundingDINO/.asset/grounded.onnx"
    shape_infered_onnx_file = "/offboard/GroundingDINO/.asset/grounded_shape_inferred.onnx"
    with torch.no_grad():
        # outputs = model(image[None], tokenized["input_ids"],
        #                 tokenized["attention_mask"], position_ids,
        #                 tokenized["token_type_ids"], text_self_attention_masks)
        image_cpu = image[None].cpu()
        input_ids_cpu = tokenized["input_ids"].cpu()
        attention_mask_cpu = tokenized["attention_mask"].cpu().to(bool)
        position_ids_cpu = position_ids.cpu()
        token_type_ids_cpu = tokenized["token_type_ids"].cpu()
        text_self_attention_masks_cpu = text_self_attention_masks.cpu()
        outputs = model_cpu(image_cpu, input_ids_cpu, attention_mask_cpu, position_ids_cpu, token_type_ids_cpu, text_self_attention_masks_cpu)
        model_cpu.export_onnx = True
        torch.onnx.export(
            model_cpu,
            f=onnx_path,
            args=(image_cpu, input_ids_cpu, attention_mask_cpu, position_ids_cpu, token_type_ids_cpu, text_self_attention_masks_cpu), #, zeros, ones),
            input_names=["img" , "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"],
            output_names=["logits", "boxes"],
            dynamic_axes=None,
            do_constant_folding=True,
            export_params=True, 
            opset_version=20)
    ######################### onnx shape inference
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print('Model was successfully converted to ONNX format.')
    # inferred_model_onnx = SymbolicShapeInference.infer_shapes(
    #     model_onnx, auto_merge=True, verbose=True)
    # onnx.save(inferred_model_onnx, shape_infered_onnx_file)
    # import pdb;pdb.set_trace()
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)
    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases

    
    return boxes_filt, pred_phrases

if __name__ == "__main__":
    # cfg
    config_file = "/offboard/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
    checkpoint_path = "/offboard/GroundingDINO/weights/groundingdino_swint_ogc.pth"  # change the path of the model
    output_dir = "/offboard/GroundingDINO/logs"
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=True)
    model_cpu = load_model(config_file, checkpoint_path, cpu_only=True)
    output_dir = "/offboard/GroundingDINO/logs"
    image_path = "/offboard/GroundingDINO/.asset/zed2.png"
    text_prompt = "bicycle. skateboard. door handle. luggage case. container. shoe box"
    
    image_pil, image = load_image(image_path)
    #export openvino
    boxes_filt, pred_phrases = export_openvino(model_cpu, model, image, text_prompt, output_dir)
    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    # import ipdb; ipdb.set_trace()
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    image_with_box.save(os.path.join(output_dir, "pred.jpg"))
    



    
