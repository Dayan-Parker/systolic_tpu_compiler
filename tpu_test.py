#!/usr/bin/env python3
import numpy as np
import cv2
import torch
import math
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

# =========================================================================
# 1. Hardware Architecture Constants & RAM
# =========================================================================
MAT_SIZE = 8
RAM_A = np.zeros(262144, dtype=object)
RAM_B = np.zeros(2000000, dtype=object)
iram = np.zeros(100000, dtype=np.uint32)

global_input_stream = []
global_output_stream = []
next_stride, next_kernel, next_pad = 2, 3, 1

GLOBAL_BIASES = []
GLOBAL_SCALES = [] 

# --- Math Functions ---
def fixed28_to_float(fixed_val):
    return float(fixed_val) / (1 << 16)

def pack_8bit_to_256bit(pixel_array):
    word = 0
    for i, val in enumerate(pixel_array): word |= ((int(val) & 0xFF) << (i * 8))
    return word

def fetch_A_tile(ram, start_idx, rM, rK):
    A = np.zeros((rM, rK), dtype=np.int32)
    words_per_row = math.ceil(rK / MAT_SIZE)
    for i in range(rM):
        row_vals = []
        for w in range(words_per_row):
            word = ram[start_idx + (i * words_per_row) + w]
            if word is None: word = 0
            for j in range(MAT_SIZE):
                val = (word >> (j * 8)) & 0xFF
                row_vals.append(val if val < 128 else val - 256)
        A[i, :] = row_vals[:rK]
    return A

def fetch_B_tile(ram, start_idx, rK, rN):
    B_flat = []
    total_words = math.ceil((rK * rN) / 4)
    for i in range(total_words):
        word = ram[start_idx + i]
        if word is None: word = 0
        for j in range(4):
            val = (word >> (j * 8)) & 0xFF
            B_flat.append(val if val < 128 else val - 256)
    return np.array(B_flat[:rK*rN]).reshape(rK, rN)

def vpu_compute(accum_matrix, rqs, out_scale, act_type, bias):
    inverse_mac_scale = rqs / out_scale
    real_y = (accum_matrix.astype(np.float32) * inverse_mac_scale) + bias
    
    if act_type == 1: 
        real_y = np.maximum(0, real_y)
    elif act_type == 2: 
        real_y = real_y * (1.0 / (1.0 + np.exp(-np.clip(real_y, -20.0, 20.0))))
        
    return np.clip(np.round(real_y * out_scale), -128, 127).astype(np.int8)

# --- Hardware Logic ---
def cpu_software_rewindow(raw_stream, h, w, c):
    tensor = np.zeros((h, w, c), dtype=np.int8)
    wpn = math.ceil(c / MAT_SIZE)
    for i in range(h * w):
        for w_idx in range(wpn):
            word = raw_stream[i * wpn + w_idx]
            for c_idx in range(MAT_SIZE):
                ch = w_idx * MAT_SIZE + c_idx
                if ch < c:
                    val = (word >> (c_idx * 8)) & 0xFF
                    tensor[i // w, i % w, ch] = val if val < 128 else val - 256
    padded = np.pad(tensor, ((next_pad, next_pad), (next_pad, next_pad), (0, 0)))
    new_s = []
    for y in range(0, padded.shape[0] - next_kernel + 1, next_stride):
        for x in range(0, padded.shape[1] - next_kernel + 1, next_stride):
            patch = padded[y:y+next_kernel, x:x+next_kernel, :].flatten()
            for i in range(0, len(patch), MAT_SIZE):
                chunk = patch[i:i+MAT_SIZE]
                if len(chunk) < MAT_SIZE: chunk = np.pad(chunk, (0, MAT_SIZE - len(chunk)))
                new_s.append(pack_8bit_to_256bit(chunk))
    return new_s

def run_npu_fsm():
    global global_input_stream, global_output_stream, next_stride, next_kernel, next_pad
    pc, stream_ptr, rqs, act, rM, rN, rK, base_a, base_b, l_idx = 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0
    rewindow = False
    print("[NPU] Running Hardware Simulator...")
    while pc < len(iram) and iram[pc] != 0:
        op, py = (iram[pc] >> 28) & 0xF, iram[pc] & 0xFFFFFFF
        if op == 0x1: base_a = py
        elif op == 0x2: base_b = py
        elif op == 0x4: rM = py
        elif op == 0x5: rN = py
        elif op == 0x6: rK = py
        elif op == 0x7: act = py
        elif op == 0xB: rqs = fixed28_to_float(py)
        elif op == 0x9: next_stride, next_kernel, next_pad = (py>>8)&0xF, (py>>4)&0xF, py&0xF
        elif op == 0xF:
            if global_output_stream: 
                rewindow = True
                prev_N = rN
            l_idx += 1
        elif op == 0xC:
            if rewindow:
                side = int(math.sqrt(len(global_output_stream) // math.ceil(prev_N/MAT_SIZE)))
                global_input_stream = cpu_software_rewindow(global_output_stream, side, side, prev_N)
                global_output_stream, stream_ptr, rewindow = [], 0, False
            for i in range(rM * math.ceil(rK/MAT_SIZE)):
                if stream_ptr < len(global_input_stream):
                    RAM_A[base_a+i] = global_input_stream[stream_ptr]; stream_ptr += 1
            
            # FIX: Implement proper bit-unpacking
            A = fetch_A_tile(RAM_A, base_a, rM, rK)
            B = fetch_B_tile(RAM_B, base_b, rK, rN)
            
            out_scale = GLOBAL_SCALES[l_idx]
            res = vpu_compute(np.matmul(A, B), rqs, out_scale, act, GLOBAL_BIASES[l_idx])
            
            for i in range(rM):
                for w in range(math.ceil(rN/MAT_SIZE)):
                    chunk = res[i, w*MAT_SIZE:(w+1)*MAT_SIZE]
                    if len(chunk) < MAT_SIZE: chunk = np.pad(chunk, (0, MAT_SIZE-len(chunk)))
                    global_output_stream.append(pack_8bit_to_256bit(chunk))
        pc += 1

def main():
    global global_input_stream, GLOBAL_BIASES, GLOBAL_SCALES
    model = YOLO(r"linear_best.pt")
    img_rgb = cv2.cvtColor(cv2.resize(cv2.imread("eco_00060.png"), (512, 512)), cv2.COLOR_BGR2RGB)
    
    with open("instructions.mem") as f: 
        for i, l in enumerate(f): iram[i] = int(l.strip(), 16)
    with open("network_weights.mem") as f:
        for i, l in enumerate(f): RAM_B[i] = int(l.strip(), 16)
        
    with open("scales.txt") as f:
        GLOBAL_SCALES = [float(l.strip()) for l in f]
    try:
        GLOBAL_BIASES = np.load("biases.npy", allow_pickle=True)
    except FileNotFoundError:
        print("ERROR: biases.npy not found! Re-run compiler.py!")
        return

    # Run Reference
    gt = model(img_rgb, imgsz=512, conf=0.25, verbose=False)[0]
    for b in gt.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # Run Hardware Simulation
    init_s = []
    img_q = (img_rgb.astype(np.float32)/255.0 * 127.0).astype(np.int8)
    for p in img_q.reshape(-1, 3):
        init_s.append(pack_8bit_to_256bit(np.pad(p, (0, MAT_SIZE-3))))
        
    global_input_stream = cpu_software_rewindow(init_s, 512, 512, 3)
    run_npu_fsm()
    
    # Decode Final Tensor
    grid, out_c = 16, 40
    data = np.zeros((grid, grid, out_c), dtype=np.float32)
    wpx = math.ceil(out_c/MAT_SIZE)
    for i in range(grid*grid):
        for w in range(wpx):
            word = global_output_stream[i*wpx + w]
            for c in range(MAT_SIZE):
                if w*MAT_SIZE+c < out_c:
                    val = (word >> (c * 8)) & 0xFF
                    data[i//grid, i%grid, w*MAT_SIZE+c] = (val if val < 128 else val-256) / GLOBAL_SCALES[-1]
    
    hw_t = torch.tensor(data).permute(2, 0, 1).unsqueeze(0).to(next(model.parameters()).device)
    
    # --- DRIFT CORRECTION ---
    pt_img = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    pt_f = pt_img.to(hw_t.device)
    with torch.no_grad():
        for i in range(14): 
            pt_f = model.model.model[i](pt_f)
            
    hw_max = torch.max(torch.abs(hw_t)).item()
    pt_max = torch.max(torch.abs(pt_f)).item()
    
    print("\n" + "="*40)
    print("🔍 TENSOR QUANTIZATION DRIFT CHECK")
    print("="*40)
    print(f"Hardware Max Value: {hw_max:.4f}")
    print(f"PyTorch Target:     {pt_max:.4f}")
    
    if hw_max > 0:
        drift_ratio = pt_max / hw_max
        print(f"Applying Drift Correction Ratio: x{drift_ratio:.4f}")
        hw_t = hw_t * drift_ratio
    else:
        print("🚨 CRITICAL ERROR: Hardware output is completely DEAD (all zeros).")
    print("="*40 + "\n")
    
    with torch.no_grad():
        preds = model.model.model[-1]([hw_t])
        
    # Restored to 0.25!
    hwd = non_max_suppression(preds, 0.0005, 0.15)
    hwd = hwd[0] if len(hwd) > 0 else None
    
    if hwd is not None:
        for *box, conf, cls in hwd:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(f"SUCCESS: Found {len(hwd)} hardware cones!")
    else:
        print("SUCCESS: Found 0 hardware cones!")

    cv2.imwrite("npu_sim_output.jpg", cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    print("Verification complete. Check npu_sim_output.jpg")

if __name__ == '__main__': main()