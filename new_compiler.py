#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2

MAT_SIZE = 8

# --- Quantization & Packing Helpers ---
def fuse_conv_and_bn(weights, bias, gamma, beta, mean, var, epsilon=1e-5):
    std = np.sqrt(var + epsilon)
    scale = gamma / std
    fused_weights = weights * scale.reshape(-1, 1, 1, 1)
    fused_bias = (bias - mean) * scale + beta
    return fused_weights, fused_bias

def quantize_to_int8(tensor):
    max_val = np.max(np.abs(tensor))
    scale = 127.0 / max_val if max_val > 0 else 1.0
    return np.round(tensor * scale).astype(np.int8), scale

def float_to_fixed28(scale_float):
    return int(round(scale_float * (1 << 16))) & 0xFFFFFFF

def pack_weights_to_hex(int8_matrix):
    flat = int8_matrix.flatten().astype(np.uint8)
    hex_words = []
    if len(flat) % 4 != 0: flat = np.pad(flat, (0, 4 - (len(flat) % 4)))
    for i in range(0, len(flat), 4):
        b0, b1, b2, b3 = int(flat[i]), int(flat[i+1]), int(flat[i+2]), int(flat[i+3])
        word = (b3 << 24) | (b2 << 16) | (b1 << 8) | b0
        hex_words.append(f"{word:08X}")
    return hex_words

def make_opcode(opcode_hex, payload_val):
    return f"{(int(opcode_hex, 16) << 28 | (int(payload_val) & 0xFFFFFFF)):08X}"

# --- Calibration using Forward Hooks ---
def calibrate_activations(model, img_path):
    print(f"[CALIBRATOR] Calibrating range using {img_path}...")
    scales = []
    
    # This "spy" function runs every time a Conv layer finishes
    def hook_fn(module, input, output):
        m = torch.max(torch.abs(output)).item()
        # Calculate scale to keep the hardware safely within -127 to 127
        scales.append(127.0 / (m * 1.2) if m > 0 else 1.0)

    hooks = []
    # Attach the spy to every compiled layer
    for name, mod in model.model.named_modules():
        if "model.14" in name: continue
        if hasattr(mod, 'conv') and hasattr(mod, 'bn'):
            hooks.append(mod.register_forward_hook(hook_fn))
            
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(cv2.resize(img, (512, 512)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).float() / 255.0
    
    with torch.no_grad():
        # Push the image through the raw PyTorch neural net
        model.model(t)
        
    for h in hooks: h.remove()
    print(f"[CALIBRATOR] Captured {len(scales)} unique layer scales.")
    return scales

# --- Compiler ---
def main_compiler(weights_path, cal_img_path):
    model = YOLO(weights_path)
    scales = calibrate_activations(model, cal_img_path)
    
    print("[COMPILER] Generating Instructions & Weight memory...")
    insts, weights, biases = [], [], []
    curr_res, addr_b, l_idx = (512, 512), 0, 0
    
    for name, mod in model.model.named_modules():
        if "model.14" in name: continue
        if not (hasattr(mod, 'conv') and hasattr(mod, 'bn')): continue
        
        f_w, f_b = fuse_conv_and_bn(mod.conv.weight.detach().cpu().numpy(), np.zeros(mod.conv.weight.shape[0]),
                                     mod.bn.weight.detach().cpu().numpy(), mod.bn.bias.detach().cpu().numpy(),
                                     mod.bn.running_mean.detach().cpu().numpy(), mod.bn.running_var.detach().cpu().numpy())
        
        biases.append(f_b)
        int8_w, sw = quantize_to_int8(f_w.transpose(2, 3, 1, 0))
        
        in_s = 127.0 if l_idx == 0 else scales[l_idx-1]
        requant = scales[l_idx] / (in_s * sw)
        
        h_in, w_in = curr_res
        s, p, k = mod.conv.stride[0], mod.conv.padding[0], mod.conv.kernel_size[0]
        h_out, w_out = ((h_in + 2*p - k)//s)+1, ((w_in + 2*p - k)//s)+1
        act = 2 if isinstance(getattr(mod, 'act', None), nn.SiLU) else (1 if isinstance(getattr(mod, 'act', None), nn.ReLU) else 0)
        
        weights.extend(pack_weights_to_hex(int8_w))
        insts.extend([make_opcode('3', MAT_SIZE), make_opcode('5', int8_w.shape[3]), make_opcode('6', int8_w.shape[0]*int8_w.shape[1]*int8_w.shape[2]),
                      make_opcode('2', addr_b), make_opcode('7', act), make_opcode('B', float_to_fixed28(requant)), make_opcode('9', (s<<8)|(k<<4)|p)])
        
        m_rem, ping = h_out*w_out, True
        while m_rem > 0:
            tile = min(m_rem, 1024)
            addr = 0 if ping else 16384
            insts.extend([make_opcode('4', tile), make_opcode('1', addr), make_opcode('8', addr), make_opcode('C', 0)])
            m_rem -= tile; ping = not ping
        
        addr_b += len(pack_weights_to_hex(int8_w))
        insts.append(make_opcode('F', 0))
        curr_res, l_idx = (h_out, w_out), l_idx + 1

    with open("instructions.mem", "w") as f: f.write("\n".join(insts))
    with open("network_weights.mem", "w") as f: f.write("\n".join(weights))
    with open("scales.txt", "w") as f: f.write("\n".join([str(s) for s in scales]))
    bias_hw_hex = []
    for l_idx, b_array in enumerate(biases):
        out_scale = scales[l_idx]
        for b_val in b_array:
            #(Bias * Out_Scale) * 65536
            b_int64 = int(round(b_val * out_scale * 65536.0)) & 0xFFFFFFFFFFFFFFFF
            bias_hw_hex.append(f"{b_int64:016X}")
            
    with open("biases.mem", "w") as f: f.write("\n".join(bias_hw_hex))
    np.save("biases.npy", np.array(biases, dtype=object))
    print(f"[COMPILER] Success! Wrote {len(insts)} instructions to .mem files.")

if __name__ == "__main__":
    main_compiler(r"linear_best.pt", "amz_00000.jpg")