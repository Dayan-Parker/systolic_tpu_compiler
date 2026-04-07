import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

# --- CONFIGURATION ---
csv_path = Path(r"runs\detect\fsoco_training\run_linear_cnn\results.csv") 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

def animate(i):
    # 1. PATH DEBUGGER
    if not csv_path.exists():
        print(f"Looking for file at: {csv_path.absolute()}")
        print(f"Waiting for {csv_path.name} to be created...", end="\r")
        return

    try:
        data = pd.read_csv(csv_path)
        if data.empty:
            return

        # Clean columns by stripping whitespace
        data.columns = [c.strip() for c in data.columns]
        epochs = data['epoch']

        ax1.clear()
        ax2.clear()

        # 2. NO MORE SAFETY NETS. If the column is wrong, throw an error!
        ax1.plot(epochs, data['train/box_loss'], label='Train Box Loss', color='orange', linestyle='--')
        ax1.plot(epochs, data['val/box_loss'], label='Val Box Loss', color='red')
        
        ax1.set_title("Box Loss (Lower is Better)")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, data['metrics/mAP50(B)'], label='mAP@50', color='blue', linewidth=2)
        ax2.plot(epochs, data['metrics/mAP50-95(B)'], label='mAP@50-95 (Strict)', color='green')
        
        ax2.set_title("Accuracy / mAP (Higher is Better)")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Precision")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        print(f"Successfully updated plot with {len(data)} epochs.", end="\r")

    except KeyError as e:
        # 3. COLUMN DEBUGGER
        print(f"\n[ERROR] Pandas could not find column {e} in the CSV!")
        print(f"Available columns are: {list(data.columns)}")
    except Exception as e:
        print(f"\n[ERROR] Plot update failed: {e}")

if __name__ == "__main__":
    print("Starting Live Monitor...")
    print("Keep this window open. Close the graph window to stop.")
    
    # Check every 5 seconds
    ani = FuncAnimation(fig, animate, interval=5000, cache_frame_data=False)
    plt.show()