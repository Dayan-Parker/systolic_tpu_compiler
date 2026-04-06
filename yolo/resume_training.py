from ultralytics import YOLO

# This line is CRITICAL on Windows to prevent infinite crashing loops
if __name__ == '__main__':
    
    # Load the model from your specific path
    # (I used the path from your code snippet)
    model = YOLO(r"D:\WUAIR\wuair_perception\perception\perception\yolo\runs\detect\fsoco_training\run_high_power\weights\last.pt")

    # Resume training
    print("Resuming training...")
    results = model.train(resume=True)