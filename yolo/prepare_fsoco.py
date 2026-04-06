import json
import shutil
from PIL import Image
import yaml
from pathlib import Path
from tqdm import tqdm
import random

# --- CONFIGURATION ---
SOURCE_DIR = Path(r"D:\WUAIR\fsoco")       
OUTPUT_DIR = Path(r"D:\WUAIR\fsoco_yolo") 
TRAIN_RATIO = 0.8               

FSOCO_CLASSES = {
    'blue_cone': 0,
    'yellow_cone': 1,
    'orange_cone': 2,
    'large_orange_cone': 3,
    'unknown_cone': -1
}

def create_dir_structure():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR) # CLEAN START
    
    for split in ['train', 'val']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x * dw, y * dh, w * dw, h * dh)

def process_dataset():
    create_dir_structure()
    
    image_paths = []
    print(f"Scanning {SOURCE_DIR} for images...")
    
    # Recursively find all images
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_paths.extend(list(SOURCE_DIR.rglob(ext)))

    random.shuffle(image_paths)
    print(f"Found {len(image_paths)} images. Processing...")

    processed_count = 0

    for img_path in tqdm(image_paths):
        # 1. Locate Annotation File
        # Go up levels to find the team folder (e.g. fsoco/amz)
        # Structure is usually: team/img/image.jpg
        team_dir = img_path.parent.parent 
        
        # Check standard annotation folder names
        ann_dir = team_dir / "ann"
        if not ann_dir.exists():
            ann_dir = team_dir / "annotations"
        
        if not ann_dir.exists():
            continue # No annotation folder found

        # Check both naming conventions: 
        # 1. image.json (standard)
        # 2. image.jpg.json (supervisely style)
        json_path = ann_dir / (img_path.stem + ".json")
        if not json_path.exists():
            json_path = ann_dir / (img_path.name + ".json")
        
        if not json_path.exists():
            continue # No matching JSON found

        # 2. Get Image Size
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except:
            continue

        # 3. Read JSON
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except:
            continue

        # 4. Extract Objects
        yolo_lines = []
        found_cone = False
        
        for obj in data.get('objects', []):
            class_name = obj.get('classTitle')
            class_id = FSOCO_CLASSES.get(class_name, -1)
            
            if class_id == -1: continue 

            points = obj.get('points', {}).get('exterior', [])
            if not points: continue

            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            min_x, max_x = min(x_vals), max(x_vals)
            min_y, max_y = min(y_vals), max(y_vals)

            bbox = convert_box((w, h), (min_x, min_y, max_x, max_y))
            yolo_lines.append(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
            found_cone = True

        if not found_cone:
            continue

        # 5. Save Data
        split = 'train' if random.random() < TRAIN_RATIO else 'val'
        
        target_img = OUTPUT_DIR / 'images' / split / img_path.name
        shutil.copy2(img_path, target_img)
        
        label_file = OUTPUT_DIR / 'labels' / split / (img_path.stem + ".txt")
        with open(label_file, 'w') as f:
            f.write("\n".join(yolo_lines))
            
        processed_count += 1

    # 6. Create YAML
    yaml_content = {
        'path': str(OUTPUT_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'blue_cone',
            1: 'yellow_cone',
            2: 'orange_cone',
            3: 'large_orange_cone'
        }
    }
    with open(OUTPUT_DIR / "fsoco.yaml", 'w') as f:
        yaml.dump(yaml_content, f)

    print(f"\nDone! Processed {processed_count} valid image/label pairs.")
    print(f"Dataset ready at: {OUTPUT_DIR / 'fsoco.yaml'}")

if __name__ == "__main__":
    process_dataset()