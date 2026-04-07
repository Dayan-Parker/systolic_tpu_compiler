import os

def calculate_hardware_footprint(mem_file_path):
    if not os.path.exists(mem_file_path):
        print(f"File not found: {mem_file_path}")
        return

    with open(mem_file_path, 'r') as f:
        # Count non-empty lines
        lines = [line for line in f if line.strip()]
    
    num_words = len(lines)
    total_bytes = num_words * 4  # 4 bytes per 32-bit word
    
    kb_size = total_bytes / 1024
    mb_size = kb_size / 1024
    
    print("="*40)
    print("🧠 NPU WEIGHT MEMORY FOOTPRINT")
    print("="*40)
    print(f"Total 32-bit Words : {num_words:,}")
    print(f"Exact Hardware Size: {total_bytes:,} Bytes")
    print(f"                     {kb_size:.2f} KB")
    print(f"                     {mb_size:.2f} MB")
    print("="*40)

if __name__ == "__main__":
    calculate_hardware_footprint("network_weights.mem")