import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ==========================================
# 1. CONFIGURATION
# ==========================================
# UPDATE THESE PATHS IF RUNNING LOCALLY
TRAIN_ROOT_DIR = "/kaggle/input/datasets/anurajktk/krackdata/Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/train"
VAL_ROOT_DIR = "/kaggle/input/datasets/anurajktk/krackdata/Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/val"
MODEL_SAVE_PATH = "best_model_phase1_remastered.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16 
LR = 0.0001
EPOCHS = 20

# 16-bit Mapping
VALUE_MAP = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}

# --- CALCULATE CLASS WEIGHTS ---
# Class Indices: 0:Back, 1:Trees, 2:Lush, 3:DryGrass, 4:DryBush, 5:Clutter, 6:Logs, 7:Rocks, 8:Land, 9:Sky
CLASS_WEIGHTS = torch.tensor([
    1.0,  # Background
    2.0,  # Trees 
    5.0,  # Lush Bush 
    2.0,  # Dry Grass 
    2.0,  # Dry Bush 
    10.0, # Clutter 
    10.0, # Logs 
    8.0,  # Rocks 
    1.0,  # Landscape 
    1.0   # Sky 
]).to(DEVICE)

# ==========================================
# 2. DATASET & UTILS
# ==========================================
class OfficialDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_dir = os.path.join(root_dir, "Color_Images")
        self.mask_dir = os.path.join(root_dir, "Segmentation")
        self.images = sorted(os.listdir(self.img_dir))
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = np.zeros_like(raw_mask, dtype=np.uint8)
        for k, v in VALUE_MAP.items(): mask[raw_mask == k] = v
        
        if self.transform: 
            aug = self.transform(image=image, mask=mask)
            image=aug['image']
            mask=aug['mask']
            
        return image, mask

def generate_post_training_report(model, device, test_img_dir, test_mask_dir, output_dir):
    print("\n" + "="*40)
    print("🚀 STARTING AUTOMATED POST-TRAINING REPORT")
    print("="*40)
    
    # 1. Setup
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
    os.makedirs(f"{output_dir}/failure_cases", exist_ok=True)
    os.makedirs(f"{output_dir}/overlays", exist_ok=True)
    
    CLASS_NAMES = ['Background', 'Trees', 'Lush Bush', 'Dry Grass', 'Dry Bush', 
                   'Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']
    CLASS_COLORS = np.array([
        [0, 0, 0], [34, 139, 34], [0, 255, 0], [210, 180, 140], [139, 90, 43],
        [128, 128, 0], [139, 69, 19], [128, 128, 128], [160, 82, 45], [135, 206, 235]
    ], dtype=np.uint8)
    VALUE_MAP_REPORT = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}

    # 2. Helpers
    def decode_mask_to_rgb(mask):
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for id, color in enumerate(CLASS_COLORS):
            rgb[mask == id] = color
        return rgb

    def create_overlay(image, mask, alpha=0.5):
        color_mask = decode_mask_to_rgb(mask)
        return cv2.addWeighted(image, 1, color_mask, alpha, 0)

    def compute_iou_score(pred, true):
        iou_scores = []
        for cls in range(10):
            inter = ((pred == cls) & (true == cls)).sum()
            union = ((pred == cls) | (true == cls)).sum()
            if union > 0: iou_scores.append(inter/union)
        return sum(iou_scores)/len(iou_scores) if iou_scores else 0

    # 3. Transform for Inference
    eval_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 4. Inference Loop
    results = []
    all_preds = []
    all_targets = []
    class_iou_totals = {i: [] for i in range(10)}
    
    # Handle DataParallel wrapper for inference
    inference_model = model
    if isinstance(model, torch.nn.DataParallel):
        inference_model = model.module
        
    inference_model.eval()
    test_files = sorted(os.listdir(test_img_dir))
    
    print(f"📸 Processing {len(test_files)} validation images...")
    for img_name in tqdm(test_files):
        img_path = os.path.join(test_img_dir, img_name)
        mask_path = os.path.join(test_mask_dir, img_name)
        if not os.path.exists(mask_path): continue

        # Load
        original_img = cv2.imread(img_path)
        if original_img is None: continue
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if raw_mask is None: continue
        
        true_mask = np.zeros_like(raw_mask, dtype=np.uint8)
        for k, v in VALUE_MAP_REPORT.items(): true_mask[raw_mask == k] = v

        # Predict
        aug = eval_transform(image=original_img)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = inference_model(aug)
            logits = torch.nn.functional.interpolate(logits, size=true_mask.shape, mode='bilinear', align_corners=False)
            pred_mask = logits.argmax(dim=1).squeeze().cpu().numpy()

        # Metrics
        score = compute_iou_score(pred_mask, true_mask)
        results.append((img_name, original_img, true_mask, pred_mask, score))

        for cls_id in range(10):
            intersection = ((pred_mask == cls_id) & (true_mask == cls_id)).sum()
            union = ((pred_mask == cls_id) | (true_mask == cls_id)).sum()
            if union > 0: class_iou_totals[cls_id].append(intersection / union)

        # Subsample for Confusion Matrix (1% pixels to save RAM)
        all_preds.append(pred_mask.flatten()[::100])
        all_targets.append(true_mask.flatten()[::100])

    # 5. Generate Outputs
    results.sort(key=lambda x: x[4])
    worst_cases = results[:5]
    best_cases = results[-5:]
    avg_iou = sum([x[4] for x in results]) / len(results)

    # A. Text Report
    with open(f"{output_dir}/final_model_report.txt", "w") as f:
        f.write(f"FINAL MODEL SCORE (Mean IoU): {avg_iou:.4f}\n\nPer-Class Breakdown:\n")
        class_avgs = []
        for i in range(10):
            c_avg = sum(class_iou_totals[i])/len(class_iou_totals[i]) if len(class_iou_totals[i]) > 0 else 0.0
            class_avgs.append(c_avg)
            f.write(f"{CLASS_NAMES[i]:<15}: {c_avg:.4f}\n")
        f.write("\nWorst Performing Images:\n")
        for item in worst_cases: f.write(f"{item[0]}: {item[4]:.4f}\n")

    # B. Bar Chart
    plt.figure(figsize=(10, 6))
    plt.bar(CLASS_NAMES, class_avgs, color=[c/255 for c in CLASS_COLORS])
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Per-Class Accuracy (Mean: {avg_iou:.2f})')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/per_class_metrics.png")
    plt.close()

    # C. Confusion Matrix
    cm = confusion_matrix(np.concatenate(all_targets), np.concatenate(all_preds), labels=range(10))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=False, cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

    # D. Save Visuals
    def save_vis_batch(case_list, subfolder):
        for img_name, img, true, pred, score in case_list:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1); plt.imshow(img); plt.title("Original"); plt.axis('off')
            plt.subplot(1, 3, 2); plt.imshow(decode_mask_to_rgb(true)); plt.title("Truth"); plt.axis('off')
            plt.subplot(1, 3, 3); plt.imshow(decode_mask_to_rgb(pred)); plt.title(f"Pred (IoU: {score:.2f})"); plt.axis('off')
            plt.savefig(f"{output_dir}/{subfolder}/{img_name}_comparison.png")
            plt.close()
            cv2.imwrite(f"{output_dir}/overlays/{img_name}_overlay.png", cv2.cvtColor(create_overlay(img, pred), cv2.COLOR_RGB2BGR))

    save_vis_batch(worst_cases, "failure_cases")
    save_vis_batch(best_cases, "visualizations")
    
    print(f"✅ REPORT GENERATED SUCCESSFULLY IN: {output_dir}")

# ==========================================
# 3. EXECUTION BLOCK (GUARDED)
# ==========================================
if __name__ == "__main__":
    # Transforms
    train_transform = A.Compose([
        A.Resize(512, 512),  
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(0.3, 0.3, 0.3, 0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2()
    ])

    train_dataset = OfficialDataset(TRAIN_ROOT_DIR, transform=train_transform)
    val_dataset = OfficialDataset(VAL_ROOT_DIR, transform=val_transform)

    # DATALOADERS are safe here because of __main__ check
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- MODEL ---
    print("🏗️ Building ResNet101 (Remastered)...")
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101", 
        encoder_weights="imagenet", 
        classes=10, 
        activation=None
    )

    if torch.cuda.device_count() > 1: 
        print(f"🔥 Detected {torch.cuda.device_count()} GPUs! Activating DataParallel...")
        model = torch.nn.DataParallel(model)

    model.to(DEVICE)

    # --- TRAINING ---
    criterion = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    print("🚀 STARTING PHASE 1 REMASTERED (Weighted Loss)...")
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = 0
        for images, masks in pbar:
            images = images.to(DEVICE); masks = masks.long().to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE); masks = masks.long().to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, MODEL_SAVE_PATH)
            print("✅ Saved Phase 1 Remastered Model!")

    print("🎉 Training Complete.")

    # --- TRIGGER THE REPORT ---
    print("Starting Analysis Module...")

    # --- ROBUST PATH CONFIGURATION ---
    try:
        # If running as a .py script, get the script's folder
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # If running in a Notebook (Kaggle), use the current working directory
        SCRIPT_DIR = os.getcwd()

    # Force the Output Folder to be inside the Script's directory
    REPORT_DIR = os.path.join(SCRIPT_DIR, "Final_Phase1_Remastered_Report")

    generate_post_training_report(model, DEVICE, os.path.join(VAL_ROOT_DIR, "Color_Images"), os.path.join(VAL_ROOT_DIR, "Segmentation"), REPORT_DIR)