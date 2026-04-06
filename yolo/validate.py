from ultralytics import YOLO

def generate_metrics():
    print("Loading the best saved weights from the interrupted run...")
    
    # Point this to the 'best.pt' file from your recent training run
    model = YOLO(r"runs\detect\fsoco_training\run_linear_hardware\weights\best.pt")

    print("Running validation and generating metrics...")
    
    metrics = model.val(
        data="fsoco.yaml", 
        imgsz=256, 
        device=0,
        plots=True,
        workers=0,  # <-- THE MAGIC BULLET FOR WINERROR 1455
        batch=16    # Keep batch size conservative
    )

    print("\n--- PERFORMANCE METRICS ---")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print("---------------------------")
    print("Done! Check the 'runs/detect/fsoco_training/run_linear_hardware/val' folder for your graphs.")

if __name__ == '__main__':
    generate_metrics()