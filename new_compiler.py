#!/usr/bin/env python3
import numpy as np
import torch
import math
from ultralytics import YOLO
import torch.nn as nn

# =========================================================================
# 1. Math & Quantization Functions
# =========================================================================
def fuse_conv_and_bn(weights, bias, gamma, beta, mean, var, epsilon=1e-5):
    std = np.sqrt(var + epsilon)
    scale = gamma / std
    scale_w = scale.reshape(-1, 1, 1, 1)
    fused_weights = weights * scale_w
    fused_bias = (bias - mean) * scale + beta
    return fused_weights, fused_bias

def quantize_to_int8(tensor):
    max_val = np.max(np.abs(tensor))
    scale = 127.0 / max_val if max_val > 0 else 1.0
    quantized_tensor = np.round(tensor * scale)
    quantized_tensor = np.clip(quantized_tensor, -128, 127).astype(np.int8)
    return quantized_tensor, scale

def float_to_fixed28(scale_float, fractional_bits=16):
    fixed_val = int(round(scale_float * (1 << fractional_bits)))
    return fixed_val & 0xFFFFFFF

def pack_weights_to_hex(int8_matrix):
    flat_weights = int8_matrix.flatten()
    hex_words = []
    remainder = len(flat_weights) % 4
    if remainder != 0:
        flat_weights = np.pad(flat_weights, (0, 4 - remainder), 'constant')
    for i in range(0, len(flat_weights), 4):
        b0, b1, b2, b3 = int(flat_weights[i]), int(flat_weights[i+1]), int(flat_weights[i+2]), int(flat_weights[i+3])
        word_32 = ((b3 & 0xFF) << 24) | ((b2 & 0xFF) << 16) | ((b1 & 0xFF) << 8) | (b0 & 0xFF)
        hex_words.append(f"{word_32:08X}")
    return hex_words

def make_opcode(opcode_hex, payload_val):
    payload_28bit = int(payload_val) & 0xFFFFFFF
    opcode_int = int(opcode_hex, 16) << 28
    return f"{(opcode_int | payload_28bit):08X}"

# =========================================================================
# 2. CISC Instruction Generators
# =========================================================================
def generate_geometry_meta(stride, kernel, padding):
    """Opcode 0x9: Tells the CPU Driver the windowing for the NEXT layer."""
    payload = (stride << 8) | (kernel << 4) | padding
    return make_opcode('9', payload)

def generate_layer_setup(N, K, sys_arr_size, start_addr_b, act_type, scale_factor, stride, kernel):
    inst = []
    inst.append(make_opcode('3', sys_arr_size))
    inst.append(make_opcode('5', N))
    inst.append(make_opcode('6', K))
    inst.append(make_opcode('2', start_addr_b))
    inst.append(make_opcode('7', act_type))
    inst.append(make_opcode('B', float_to_fixed28(scale_factor)))
    inst.append(generate_geometry_meta(stride, kernel, 1)) # Default Pad=1
    return inst

def generate_tile_execution(M_tile, start_addr_a, start_addr_c):
    inst = []
    inst.append(make_opcode('4', M_tile))
    inst.append(make_opcode('1', start_addr_a))
    inst.append(make_opcode('8', start_addr_c))
    inst.append(make_opcode('C', 0)) # RUN_LAYER
    return inst

# =========================================================================
# 3. Network Extraction
# =========================================================================
def process_pytorch_module(module, current_spatial_dim, verbose=True):
    if hasattr(module, 'conv') and hasattr(module, 'bn'):
        weights = module.conv.weight.detach().cpu().numpy()
        bias = np.zeros(weights.shape[0], dtype=np.float32)
        gamma, beta = module.bn.weight.detach().cpu().numpy(), module.bn.bias.detach().cpu().numpy()
        mean, var = module.bn.running_mean.detach().cpu().numpy(), module.bn.running_var.detach().cpu().numpy()

        fused_w, fused_b = fuse_conv_and_bn(weights, bias, gamma, beta, mean, var)
        int8_weights, scale_factor = quantize_to_int8(fused_w)
        packed_hex_weights = pack_weights_to_hex(int8_weights)

        out_ch, in_ch, k_h, k_w = int8_weights.shape
        stride = module.conv.stride[0]
        padding = module.conv.padding[0]

        H_in, W_in = current_spatial_dim
        H_out = ((H_in + 2 * padding - k_h) // stride) + 1
        W_out = ((W_in + 2 * padding - k_w) // stride) + 1

        M, N, K = H_out * W_out, out_ch, in_ch * k_h * k_w
        act_type = 2 if isinstance(getattr(module, 'act', None), nn.SiLU) else (1 if isinstance(getattr(module, 'act', None), nn.ReLU) else 0)

        if verbose: print(f"CONV -> M:{M} N:{N} K:{K} | Spatial: {H_out}x{W_out} | Stride: {stride}")

        # We now return stride and kernel size (k_h) as well
        return 'conv', (M, N, K), (H_out, W_out), act_type, scale_factor, int8_weights, packed_hex_weights, stride, k_h

    return 'ignore', None, current_spatial_dim, 0, 1.0, None, None, 0, 0

# =========================================================================
# 4. Main Compiler Execution
# =========================================================================
def main(weights_path, input_resolution, systolic_arr_size, ram_a_size_bytes):
    print(f"Loading PyTorch model from {weights_path}...")
    model = YOLO(weights_path)
    all_instructions, all_hex_weights = [], []
    current_spatial_dim = input_resolution
    addr_b_global_offset = 0
    PING_ADDR_A, PONG_ADDR_A = 0x0000, (ram_a_size_bytes // 2) // 8
    layer_counter = 1

    for idx, module in enumerate(model.model.modules()):
        res = process_pytorch_module(module, current_spatial_dim, verbose=True)
        layer_type, dimensions, current_spatial_dim, act_type, scale_factor, _, hex_weights, stride, kernel = res

        if layer_type == 'conv':
            M, N, K = dimensions
            all_hex_weights.extend(hex_weights)

            # Fixed call: passing stride and kernel
            setup_inst = generate_layer_setup(N, K, systolic_arr_size, addr_b_global_offset, act_type, scale_factor, stride, kernel)
            all_instructions.extend(setup_inst)

            max_m_per_tile = ((ram_a_size_bytes // 2) // K // systolic_arr_size) * systolic_arr_size
            if max_m_per_tile == 0: raise MemoryError(f"Layer {layer_counter} (K={K}) is too thick for RAM_A!")

            m_remaining, use_ping = M, True
            while m_remaining > 0:
                current_m_tile = min(m_remaining, max_m_per_tile)
                addr_a = PING_ADDR_A if use_ping else PONG_ADDR_A
                all_instructions.extend(generate_tile_execution(current_m_tile, addr_a, addr_a)) # Writing back to A for demo
                m_remaining -= current_m_tile
                use_ping = not use_ping

            addr_b_global_offset += len(hex_weights)
            all_instructions.append(make_opcode('F', 0)) # HALT
            layer_counter += 1

    with open("instructions.mem", "w") as f: f.write("\n".join(all_instructions))
    with open("network_weights.mem", "w") as f: f.write("\n".join(all_hex_weights))
    print(f"\nSuccess: Wrote {len(all_instructions)} instructions.")

if __name__ == "__main__":
    main("best.pt", (240, 240), 32, 262144)
