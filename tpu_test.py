#!/usr/bin/env python3
import numpy as np
import cv2
import torch
import math
from ultralytics.utils.nms import non_max_suppression

# =========================================================================
# 1. Hardware Architecture Constants
# =========================================================================
MAT_SIZE = 32
INPUT_WIDTH = 8
RAM_A = np.zeros(262144, dtype=object)
RAM_B = np.zeros(2000000, dtype=object)
iram = np.zeros(100000, dtype=np.uint32)

global_input_stream = []
global_output_stream = []
stream_ptr = 0

# Geometry State for CPU Rewindowing (Updated by Opcode 0x9)
next_stride, next_kernel, next_pad = 2, 3, 1

def pack_8bit_to_256bit(pixel_array):
    word = 0
    for i in range(len(pixel_array)):
        val = int(pixel_array[i]) & 0xFF
        word |= (val << (i * 8))
    return word

def fetch_tile(ram, start_idx, rows, cols):
    """Unpacks hardware words into a NumPy matrix for REAL math simulation."""
    tile = np.zeros((rows, cols), dtype=np.int32)
    for i in range(rows):
        if start_idx + i < len(ram):
            word = ram[start_idx + i]
            if word is None or word == 0: continue
            for j in range(cols):
                val = (word >> (j * 8)) & 0xFF
                tile[i, j] = val if val < 128 else val - 256
    return tile

def vpu_compute(accum_matrix, scale, act_type):
    """Simulates the Vector Processing Unit (Scaling + Activation)."""
    # Fixed-point scale: (Sum * Scale) >> 16
    scaled = (accum_matrix.astype(np.int64) * scale) >> 16
    clamped = np.clip(scaled, -128, 127).astype(np.int8)

    if act_type == 1: # ReLU
        clamped[clamped < 0] = 0
    elif act_type == 2: # SiLU (Approximate Hardware LUT)
        x = clamped.astype(np.float32) / 16.0
        silu = clamped.astype(np.float32) * (1.0 / (1.0 + np.exp(-x)))
        clamped = np.clip(np.round(silu), -128, 127).astype(np.int8)

    return clamped

# =========================================================================
# 2. Host CPU: Software im2col Re-Windowing
# =========================================================================
def cpu_software_rewindow(raw_stream, h, w, c):
    """Host CPU simulation of im2col re-windowing between HALTS."""
    print(f"   [CPU INTERRUPT] Re-windowing {h}x{w}x{c} -> Stride {next_stride}")
    tensor = np.zeros((h, w, c), dtype=np.int8)

    words_per_n = math.ceil(c / MAT_SIZE)
    pixel_idx = 0

    for i in range(0, len(raw_stream), words_per_n):
        if pixel_idx >= h * w: break
        curr_h, curr_w = pixel_idx // w, pixel_idx % w

        channels_extracted = 0
        for w_idx in range(words_per_n):
            if i + w_idx >= len(raw_stream): break
            word = raw_stream[i + w_idx]
            for c_idx in range(MAT_SIZE):
                if channels_extracted < c:
                    val = (word >> (c_idx * 8)) & 0xFF
                    # Signed Int8 conversion to prevent OverflowError
                    signed_val = val if val < 128 else val - 256
                    tensor[curr_h, curr_w, channels_extracted] = signed_val
                    channels_extracted += 1
        pixel_idx += 1

    k, s, p = next_kernel, next_stride, next_pad
    padded = np.pad(tensor, ((p, p), (p, p), (0, 0)), mode='constant')
    new_stream = []

    for y in range(0, padded.shape[0] - k + 1, s):
        for x in range(0, padded.shape[1] - k + 1, s):
            patch = padded[y:y+k, x:x+k, :].flatten()
            pad_len = (MAT_SIZE - (len(patch) % MAT_SIZE)) % MAT_SIZE
            if pad_len > 0:
                patch = np.pad(patch, (0, pad_len), mode='constant')
            for i in range(0, len(patch), MAT_SIZE):
                new_stream.append(pack_8bit_to_256bit(patch[i:i+MAT_SIZE]))

    return new_stream

# =========================================================================
# 3. NPU FSM Simulator (The "Real" Math Engine)
# =========================================================================
def run_npu_fsm():
    global global_input_stream, global_output_stream, stream_ptr, next_stride, next_kernel
    pc = 0
    base_a, base_b, reg_M, reg_N, reg_K = 0, 0, 0, 0, 0
    act_type, scale_factor = 0, 0

    print("[NPU] Execution Started...")
    while pc < len(iram) and iram[pc] != 0:
        inst = iram[pc]
        op, payload = (inst >> 28) & 0xF, inst & 0xFFFFFFF

        if op == 0x1: base_a = payload
        elif op == 0x2: base_b = payload
        elif op == 0x4: reg_M = payload
        elif op == 0x5: reg_N = payload
        elif op == 0x6: reg_K = payload
        elif op == 0x7: act_type = payload
        elif op == 0xB: scale_factor = payload
        elif op == 0x9:
            next_stride = (payload >> 8) & 0xF
            next_kernel = (payload >> 4) & 0xF
        elif op == 0xF: # HALT
            if len(global_output_stream) == 0:
                pc += 1
                continue

            words_per_n = math.ceil(reg_N / MAT_SIZE)
            total_pixels = len(global_output_stream) // words_per_n
            side = int(math.sqrt(total_pixels))

            global_input_stream = cpu_software_rewindow(global_output_stream, side, side, reg_N)
            global_output_stream, stream_ptr = [], 0

        elif op == 0xC: # RUN_LAYER
            # 1. DMA Input
            words_per_m = math.ceil(reg_K / MAT_SIZE)
            for i in range(reg_M * words_per_m):
                RAM_A[base_a + i] = global_input_stream[stream_ptr] if stream_ptr < len(global_input_stream) else 0
                stream_ptr += 1

            # 2. THE REAL MATH (Simulating the Systolic Array MACs)
            A_mat = fetch_tile(RAM_A, base_a, reg_M, reg_K)
            B_mat = fetch_tile(RAM_B, base_b, reg_K, reg_N)
            raw_accum = np.matmul(A_mat, B_mat)

            # 3. VPU Scaling and Activation
            activated = vpu_compute(raw_accum, scale_factor, act_type)

            # 4. DMA Output (Correctly packing matching the N dimension)
            words_per_n = math.ceil(reg_N / MAT_SIZE)
            for i in range(reg_M):
                for w in range(words_per_n):
                    chunk = activated[i, w*MAT_SIZE : (w+1)*MAT_SIZE]
                    if len(chunk) < MAT_SIZE:
                        chunk = np.pad(chunk, (0, MAT_SIZE - len(chunk)), mode='constant')
                    global_output_stream.append(pack_8bit_to_256bit(chunk))
        pc += 1

# =========================================================================
# 4. YOLO Head Decoder
# =========================================================================
def yolo_head_decoder(final_stream, num_classes=4):
    print(f"\n[HOST CPU] Decoding YOLO Head ({len(final_stream)} words)...")
    raw_values = []

    # The final layer outputs 8 channels (4 coords + 4 classes)
    out_c = 4 + num_classes

    for word in final_stream:
        for c_idx in range(out_c): # Only extract the valid channels!
            val = (word >> (c_idx * 8)) & 0xFF
            val_signed = val if val < 128 else val - 256
            raw_values.append(float(val_signed) / 127.0)

    num_anchors = 1185 # Standard for 240x240 Pico
    total_elements = out_c * num_anchors

    if len(raw_values) < total_elements:
        print(f"   [!] Stream short: {len(raw_values)} < {total_elements}. Padding...")
        raw_values.extend([0.0] * (total_elements - len(raw_values)))

    preds = torch.tensor(raw_values[:total_elements]).view(1, out_c, -1)

    # Run PyTorch NMS
    results = non_max_suppression(preds, conf_thres=0.1, iou_thres=0.45)
    return results[0] if len(results) > 0 else None

# =========================================================================
# 5. Main Execution
# =========================================================================
def main():
    global global_input_stream

    try:
        with open("instructions.mem", "r") as f:
            for i, l in enumerate(f): iram[i] = int(l.strip(), 16)
        with open("network_weights.mem", "r") as f:
            for i, l in enumerate(f): RAM_B[i] = int(l.strip(), 16)
    except FileNotFoundError:
        print("Error: Run your compiler first to generate .mem files!")
        return

    # Image Prep: YOLO expects normalized inputs.
    # We map [0, 255] RGB to [-128, 127] Int8 for the hardware.
    img = cv2.imread("amz_00000.jpg")
    img_rgb = cv2.cvtColor(cv2.resize(img, (240, 240)), cv2.COLOR_BGR2RGB)
    img_int8 = np.clip(img_rgb.astype(np.int16) - 128, -128, 127).astype(np.int8)

    flat = img_int8.flatten()
    # Pad flat array to ensure it's a multiple of MAT_SIZE
    if len(flat) % MAT_SIZE != 0:
        flat = np.pad(flat, (0, MAT_SIZE - (len(flat) % MAT_SIZE)))

    global_input_stream = [pack_8bit_to_256bit(flat[i:i+MAT_SIZE]) for i in range(0, len(flat), MAT_SIZE)]

    # Start the engine!
    run_npu_fsm()

    detections = yolo_head_decoder(global_output_stream if len(global_output_stream) > 0 else global_input_stream)

    if detections is not None and len(detections) > 0:
        print(f"Success! Found {len(detections)} objects.")
        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        print("No objects detected above confidence threshold.")

    cv2.imwrite("npu_sim_output.jpg", cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)) # Convert back for saving
    print("Simulation Complete. Output saved as npu_sim_output.jpg.")

if __name__ == '__main__':
    main()
