import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import sys

# --- CONFIGURATION ---
# POINT THIS TO YOUR CURRENT 512x512 TRAINING RUN!
csv_path = Path(r"runs\detect\fsoco_training\run_linear_hardware\results.csv") 

# Set up the figure and axes outside the loop so they don't spawn multiple windows
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

def animate(i):
    if not csv_path.exists():
        print(f"Waiting for {csv_path} to be created...", end="\r")
        return

    try:
        # Read the latest data
        data = pd.read_csv(csv_path)
        data.columns = [c.strip() for c in data.columns]
        epochs = data['epoch']

        # Clear the previous lines
        ax1.clear()
        ax2.clear()

        # PLOT 1: LOSS (Lower is better)
        ax1.plot(epochs, data['train/box_loss'], label='Train Box Loss', color='orange', linestyle='--')
        ax1.plot(epochs, data['val/box_loss'], label='Val Box Loss', color='red')
        ax1.set_title("Box Loss (Lower is Better)")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # PLOT 2: ACCURACY (Higher is better)
        if 'metrics/mAP50(B)' in data.columns:
            ax2.plot(epochs, data['metrics/mAP50(B)'], label='mAP@50', color='blue', linewidth=2)
            ax2.plot(epochs, data['metrics/mAP50-95(B)'], label='mAP@50-95 (Strict)', color='green')
        
        ax2.set_title("Accuracy / mAP (Higher is Better)")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Precision")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
    except Exception as e:
        # Prevents a crash if Python tries to read the CSV at the exact millisecond YOLO is writing to it
        pass 

if __name__ == "__main__":
    print(f"Starting Live Monitor for: {csv_path}")
    print("Keep this window open. Close the graph window to stop.")
    
    # interval=10000 means it checks the CSV every 10 seconds (10,000 milliseconds)
    ani = FuncAnimation(fig, animate, interval=10000, cache_frame_data=False)
    
    plt.tight_layout()
    plt.show()