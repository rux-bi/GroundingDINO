import tensorrt as trt
import numpy as np
import os
import sys

# Create a TensorRT logger with more verbose output
logger = trt.Logger(trt.Logger.VERBOSE)

# Path to your engine file (not profile file)
engine_path = "/home/admin/.cache/huggingface/hub/models--dakey--GDinoTRT/gdino_trt_fp16.engine"
# engine_path = "/offboard/GroundingDINO/.asset/trt_cache/TensorrtExecutionProvider_TRTKernel_graph_main_graph_16549105544916672165_0_0_fp16_sm89.engine"

# or possibly with .plan extension if that's how it was saved

# Check if the engine file exists
if not os.path.exists(engine_path):
    print(f"Engine file not found: {engine_path}")
    print("Please make sure you're pointing to a valid TensorRT engine file (.engine or .plan)")
    exit(1)

try:
    # Print TensorRT version information
    print(f"TensorRT version: {trt.__version__}")
    
    # Create a runtime
    runtime = trt.Runtime(logger)

    # Load the engine from the file
    print(f"Loading engine from: {engine_path}")
    print(f"File size: {os.path.getsize(engine_path)} bytes")
    
    with open(engine_path, "rb") as f:
        engine_data = f.read()
        print(f"Read {len(engine_data)} bytes of engine data")
        
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        print("Failed to deserialize engine. Check TensorRT version compatibility.")
        exit(1)

    # Get information about the engine
    print(f"\nEngine has {engine.num_optimization_profiles} optimization profile(s)")
    print(f"Engine has {engine.num_layers} layers")
    
    # In TensorRT 10.x, we need to create an execution context to access bindings
    context = engine.create_execution_context()
    
    # Get the number of I/O tensors
    num_io_tensors = engine.num_io_tensors
    print(f"Engine has {num_io_tensors} I/O tensors")
    # print(f"Engine max batch size: {engine.max_batch_size}")
    
    # Get information about all I/O tensors
    print("\nAll I/O tensors:")
    for i in range(num_io_tensors):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        
        print(f"  Tensor {i}: '{name}'")
        print(f"    Is input: {is_input}")
        print(f"    Shape: {shape}")
        print(f"    Data type: {dtype}")
    # Get information about the inputs by profile
    print("\nEngine inputs by profile:")
    for profile_idx in range(engine.num_optimization_profiles):
        print(f"\nProfile {profile_idx}:")
        context.set_optimization_profile_async(profile_idx, 0)
        
        try:
            for i in range(num_io_tensors):
                name = engine.get_tensor_name(i)
                
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    shape = engine.get_tensor_shape(name)
                    dtype = engine.get_tensor_dtype(name)
                    
                    try:
                        # Get the optimization profile information using context methods instead
                        # TensorRT 10.x uses different methods to access profile shapes
                        print(f"  Input '{name}':")
                        print(f"    Current shape: {shape}")
                        print(f"    Data type: {dtype}")
                        
                        # Try to get profile dimensions using context methods
                        try:
                            
                            min_dims = engine.get_tensor_profile_shape(name, profile_idx)[0]
                            opt_dims = engine.get_tensor_profile_shape(name, profile_idx)[1]
                            max_dims = engine.get_tensor_profile_shape(name, profile_idx)[2]
                            print(f"    Min shape: {min_dims}")
                            print(f"    Opt shape: {opt_dims}")
                            print(f"    Max shape: {max_dims}")
                            # import pdb; pdb.set_trace()
                        except AttributeError:
                            # If context.get_tensor_profile_shape is not available, try alternative methods
                            print(f"    Note: Profile shape information not available with this TensorRT version")
                    except Exception as profile_error:
                        print(f"  Error getting profile info for '{name}': {profile_error}")
        except Exception as binding_error:
            print(f"  Error processing tensors for profile {profile_idx}: {binding_error}")

except Exception as e:
    print(f"Error loading engine: {str(e)}")
    import traceback
    traceback.print_exc()