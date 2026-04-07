import numpy as np
import torch
import math
import struct
from ultralytics import YOLO
import torch.nn as nn

# =========================================================================
# 1. Math & Quantization Functions
# =========================================================================
def fuse_conv_and_bn(weights, bias, gamma, beta, mean, var, epsilon=1e-5):
    """Fuses BatchNorm parameters into Convolution weights and biases."""
    std = np.sqrt(var + epsilon)
    scale = gamma / std

    # Reshape scale to (out_channels, 1, 1, 1) so Numpy knows how to multiply it
    # against the (out_channels, in_channels, H, W) weight tensor.
    scale_w = scale.reshape(-1, 1, 1, 1)

    fused_weights = weights * scale_w

    # Bias is 1D (out_channels,), so standard 1D multiplication works fine here
    fused_bias = (bias - mean) * scale + beta

    return fused_weights, fused_bias

def quantize_to_int8(tensor):
    """Quantizes a floating-point tensor to 8-bit signed integers."""
    max_val = np.max(np.abs(tensor))
    scale = 127.0 / max_val if max_val > 0 else 1.0
    quantized_tensor = np.round(tensor * scale)
    quantized_tensor = np.clip(quantized_tensor, -128, 127).astype(np.int8)
    return quantized_tensor, scale

def float_to_fixed28(scale_float, fractional_bits=16):
    """Converts a float to a 28-bit fixed-point integer for the hardware payload."""
    fixed_val = int(round(scale_float * (1 << fractional_bits)))
    # Mask to 28 bits to prevent overflow into the opcode section
    return fixed_val & 0xFFFFFFF

# =========================================================================
# 2. Hardware Formatting Functions
# =========================================================================
def pack_weights_to_hex(int8_matrix):
    """Packs a 2D numpy array of int8s into a list of 32-bit hex strings."""
    flat_weights = int8_matrix.flatten()
    hex_words = []

    # Pad with zeros if the total number of weights isn't a multiple of 4
    remainder = len(flat_weights) % 4
    if remainder != 0:
        flat_weights = np.pad(flat_weights, (0, 4 - remainder), 'constant')

    for i in range(0, len(flat_weights), 4):
        b0 = int(flat_weights[i])   & 0xFF
        b1 = int(flat_weights[i+1]) & 0xFF
        b2 = int(flat_weights[i+2]) & 0xFF
        b3 = int(flat_weights[i+3]) & 0xFF

        word_32 = (b3 << 24) | (b2 << 16) | (b1 << 8) | b0
        hex_words.append(f"{word_32:08X}")

    return hex_words

def make_opcode(opcode_hex, payload_val):
    """Packs a 4-bit hex opcode and a 28-bit payload into a 32-bit string."""
    payload_28bit = int(payload_val) & 0xFFFFFFF
    opcode_int = int(opcode_hex, 16) << 28
    instruction = opcode_int | payload_28bit
    return f"{instruction:08X}"

def generate_opcode(M, N, K, sys_arr_size, start_addr_a, start_addr_b, start_addr_c, activation_type, scale_factor):
    """Generates the tiled instruction sequence for an M x K multiplied by K x N matrix."""
    instructions = []

    # 0x3: SET_MAT_SIZE
    instructions.append(make_opcode('3', sys_arr_size))

    # 0xB: SET_SCALE (Load the fixed-point scale factor into the VPU)
    fixed_scale = float_to_fixed28(scale_factor)
    instructions.append(make_opcode('B', fixed_scale))

    # Loop 1: Move down the Rows of Matrix A (and Matrix C)
    for m in range(0, M, sys_arr_size):

        # Loop 2: Move across the Columns of Matrix B (and Matrix C)
        for n in range(0, N, sys_arr_size):

            addr_c = start_addr_c + (m * N) + n
            instructions.append(make_opcode('8', addr_c))

            # 0xA: SET_ACTIVATION (0=Linear, 1=ReLU, 2=SiLU, 3=Sigmoid)
            instructions.append(make_opcode('A', activation_type))

            # Loop 3: The Accumulation Loop (Walking across K)
            for step, k in enumerate(range(0, K, sys_arr_size)):

                addr_a = start_addr_a + (m * K) + k
                addr_b = start_addr_b + (k * N) + n

                instructions.append(make_opcode('1', addr_a))
                instructions.append(make_opcode('2', addr_b))

                # 0x6: CLEAR_PE
                if step == 0:
                    instructions.append(make_opcode('6', 1))
                else:
                    instructions.append(make_opcode('6', 0))

                # 0x7: ACCUM_CTRL
                is_last_step = (k + sys_arr_size >= K)

                if step == 0 and not is_last_step:
                    instructions.append(make_opcode('7', 1)) # First tile
                elif is_last_step:
                    instructions.append(make_opcode('7', 2)) # Last tile (Push to VPU)
                else:
                    instructions.append(make_opcode('7', 3)) # Middle tiles

                # 0xF: RUN_MATRIX
                instructions.append(make_opcode('F', 0))

    # 0xE: HALT (After the entire M x N x K volume is processed)
    instructions.append(make_opcode('E', 0))
    return instructions

# =========================================================================
# 3. Network Extraction & Utilities
# =========================================================================
def track_and_print_size(int8_weights_list, model_name="Network"):
    """Calculates and prints the size of the total compressed weights."""
    total_bytes = sum([w.size for w in int8_weights_list])
    size_kb = total_bytes / 1024.0
    size_mb = size_kb / 1024.0

    print("\n" + "="*50)
    print(f" TOTAL {model_name} INT8 WEIGHT SIZE:")
    print(f" {total_bytes} Bytes")
    print(f" {size_kb:.2f} KB")
    print(f" {size_mb:.2f} MB")
    print("="*50 + "\n")

def process_pytorch_module(module, current_spatial_dim, verbose=True):
    """Extracts weights, computes spatial dimensions, and determines activation."""
    if not (hasattr(module, 'conv') and hasattr(module, 'bn')):
        return None, current_spatial_dim, 0, 1.0, None, None

    # Extract FP32 parameters
    weights = module.conv.weight.detach().cpu().numpy()
    bias = np.zeros(weights.shape[0], dtype=np.float32)
    gamma = module.bn.weight.detach().cpu().numpy()
    beta = module.bn.bias.detach().cpu().numpy()
    mean = module.bn.running_mean.detach().cpu().numpy()
    var = module.bn.running_var.detach().cpu().numpy()

    # Fuse and Quantize
    fused_w, fused_b = fuse_conv_and_bn(weights, bias, gamma, beta, mean, var)
    int8_weights, scale_factor = quantize_to_int8(fused_w)
    packed_hex_weights = pack_weights_to_hex(int8_weights)

    # Compute full true tiling dimensions
    out_channels, in_channels, k_h, k_w = int8_weights.shape
    stride = module.conv.stride[0]
    padding = module.conv.padding[0]

    H_in, W_in = current_spatial_dim
    H_out = ((H_in + 2 * padding - k_h) // stride) + 1
    W_out = ((W_in + 2 * padding - k_w) // stride) + 1
    new_spatial_dim = (H_out, W_out)

    M = H_out * W_out
    N = out_channels
    K = in_channels * k_h * k_w

    # Determine Activation Type
    act_type = 0 # Linear
    if hasattr(module, 'act'):
        if isinstance(module.act, nn.SiLU):
            act_type = 2
        elif isinstance(module.act, nn.ReLU):
            act_type = 1

    if verbose:
        print(f"Layer M:{M} N:{N} K:{K} | Output Spatial: {H_out}x{W_out} | Act Type: {act_type}")

    dimensions = (M, N, K)
    return dimensions, new_spatial_dim, act_type, scale_factor, int8_weights, packed_hex_weights

# =========================================================================
# 4. Main Compiler Execution
# =========================================================================
def main(weights_path, input_resolution, systolic_arr_size, mem_bank_a, mem_bank_b, mem_bank_c, verbose=True):
    print(f"Loading PyTorch model from {weights_path}...")
    model = YOLO(weights_path)

    all_instructions = []
    all_hex_weights = []
    all_int8_arrays = []

    addr_a = mem_bank_a
    addr_b = mem_bank_b
    addr_c = mem_bank_c

    current_spatial_dim = input_resolution

    for idx, module in enumerate(model.model.modules()):
        if verbose:
            print(f"\n--- Processing Layer {idx} ---")

        dimensions, current_spatial_dim, act_type, scale_factor, int8_weights, hex_weights = process_pytorch_module(
            module, current_spatial_dim, verbose
        )

        if dimensions is not None:
            M, N, K = dimensions

            all_int8_arrays.append(int8_weights)
            all_hex_weights.extend(hex_weights)

            temp_inst = generate_opcode(
                M=M, N=N, K=K,
                sys_arr_size=systolic_arr_size,
                start_addr_a=addr_a,
                start_addr_b=addr_b,
                start_addr_c=addr_c,
                activation_type=act_type,
                scale_factor=scale_factor
            )

            all_instructions.extend(temp_inst)

            # Update memory pointers for the next layer
            addr_b += len(hex_weights)

            # Ping-Pong Memory logic for Layer inputs/outputs
            addr_a = addr_c
            addr_c = mem_bank_a if addr_c == mem_bank_c else mem_bank_c

    track_and_print_size(all_int8_arrays, model_name=weights_path)

    with open("execution_microcode.mem", "w") as f:
        f.write("\n".join(all_instructions))
    if verbose: print(f"Successfully wrote {len(all_instructions)} instructions.")

    with open("network_weights.mem", "w") as f:
        f.write("\n".join(all_hex_weights))
    if verbose: print(f"Successfully wrote {len(all_hex_weights)} packed 32-bit words.")

    return 0

if __name__ == "__main__":
    # Test Parameters
    NETWORK_PATH = "best.pt"
    IMAGE_RES = (240, 240) # (Height, Width)
    SYS_ARRAY_DIM = 16

    # Base Memory Addresses (Word Aligned)
    RAM_A_BASE = 0x00000
    RAM_B_BASE = 0x00000
    RAM_C_BASE = 0x10000 # Ping-Pong secondary bank

    main(
        weights_path=NETWORK_PATH,
        input_resolution=IMAGE_RES,
        systolic_arr_size=SYS_ARRAY_DIM,
        mem_bank_a=RAM_A_BASE,
        mem_bank_b=RAM_B_BASE,
        mem_bank_c=RAM_C_BASE,
        verbose=True
    )
