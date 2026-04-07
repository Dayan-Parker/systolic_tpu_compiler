from ultralytics import YOLO
import torch

def train_fsoco_model():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: Running on CPU! Check your PyTorch install.")

    # 1. LOAD THE CUSTOM LINEAR YAML!
    # Do not pass a .pt file here. We must build the custom architecture.
    model = YOLO("yolo/yolo_linear_pico.yaml") 

    results = model.train(
        data="yolo/fsoco.yaml",
        epochs=120,
        
        # 2. CRITICAL HARDWARE FIX: Must be a multiple of 32!
        # 256x256 prevents fractional pixel errors in the systolic array.
        imgsz=512,              
        
        batch=16,               
        device=0,               
        workers=10,
        hsv_h=0.015,            
        hsv_s=0.7,              
        hsv_v=0.4,              
        degrees=10.0,           
        translate=0.1,          
        scale=0.5,              
        mosaic=1.0,             
        erasing=0.4,            
        
        project="fsoco_training",
        name="run_linear_cnn",
        exist_ok=True,
        plots=True
    )

    print("Training Complete. Exporting to standard PyTorch weights...")
    
    # Export to standard formats (Optional, since we just need the best.pt file)
    model.export(format="onnx", imgsz=[512, 512], opset=12) 

if __name__ == '__main__':
    train_fsoco_model()