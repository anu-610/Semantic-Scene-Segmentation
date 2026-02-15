import sys
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. IMPORT UTILS & DINO HEAD DEFINITION
# ==========================================
try:
    import model_utils
    print("✅ model_utils loaded successfully.")
except ImportError:
    print("❌ CRITICAL ERROR: Could not find 'model_utils.py'.")
    sys.exit(1)

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 128, 7, padding=3), nn.BatchNorm2d(128), nn.GELU())
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, 7, padding=3, groups=128), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 1), nn.GELU()
        )
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        # Reconstruct the spatial grid from tokens
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.classifier(self.block(self.stem(x)))

# ==========================================
# 2. CONFIGURATION
# ==========================================
try:
    current_script_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_script_path = os.getcwd()

TEST_IMG_DIR = "/kaggle/input/datasets/devil610/testdata-krackhack/Offroad_Segmentation_testImages/Color_Images"
TEST_MASK_DIR = "/kaggle/input/datasets/devil610/testdata-krackhack/Offroad_Segmentation_testImages/Segmentation"
OUTPUT_DIR = os.path.join(current_script_path, "Submission_Results_DINOv2")

# --- MODEL PATH (FAILSAFE) ---
MODEL_PATH = model_utils.get_model_path(
    "best_model_dino.pth", 
    "/kaggle/input/models/dino-vit-l14/pytorch/default/1/best_model.pth"
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = (616, 616) # Must be multiple of 14 for ViT-L/14

VALUE_MAP = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}
CLASS_NAMES = ['Background', 'Trees', 'Lush Bush', 'Dry Grass', 'Dry Bush', 'Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']
CLASS_COLORS = np.array([[0,0,0], [34,139,34], [0,255,0], [210,180,140], [139,90,43], [128,128,0], [139,69,19], [128,128,128], [160,82,45], [135,206,235]], dtype=np.uint8)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/failure_cases", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/overlays", exist_ok=True)

# ==========================================
# 3. UTILS
# ==========================================
def compute_official_iou(pred, true, num_classes=10):
    iou_list = []
    for c in range(num_classes):
        i = ((pred == c) & (true == c)).sum()
        u = ((pred == c) | (true == c)).sum()
        if u > 0: iou_list.append(i/u)
    return sum(iou_list)/len(iou_list) if iou_list else 0.0

def decode_mask(mask):
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, c in enumerate(CLASS_COLORS): rgb[mask == i] = c
    return rgb

# ==========================================
# 4. LOAD DINO BACKBONE & HEAD
# ==========================================
print(f"⏳ Loading DINOv2 Backbone...")
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(DEVICE).eval()

# Determine embed_dim and patch grid
token_h, token_w = IMG_SIZE[0]//14, IMG_SIZE[1]//14
embed_dim = 1024 # Standard for Vit-L/14

print(f"⏳ Loading DINOv2 Segmentation Head from {MODEL_PATH}...")
head = SegmentationHeadConvNeXt(embed_dim, 10, token_w, token_h).to(DEVICE)

# PyTorch 2.6 Fix: weights_only=False
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
if 'module.' in list(state_dict.keys())[0]:
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
head.load_state_dict(state_dict)
head.eval()

transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ==========================================
# 5. INFERENCE LOOP
# ==========================================
print("🚀 DINOv2 Inference Started...")
results, all_preds, all_targets, class_ious = [], [], [], {i: [] for i in range(10)}

test_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])

for img_name in tqdm(test_files):
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    mask_path = os.path.join(TEST_MASK_DIR, img_name)
    if not os.path.exists(mask_path): continue

    img_bgr = cv2.imread(img_path)
    original_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    true_mask = np.zeros_like(raw_mask, dtype=np.uint8)
    for k, v in VALUE_MAP.items(): true_mask[raw_mask == k] = v

    tensor = transform(image=original_img)["image"].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # DINOv2 Feature Extraction
        features = backbone.forward_features(tensor)["x_norm_patchtokens"]
        # Pass through ConvNeXt Head
        logits = head(features)
        # Interpolate back to original resolution
        logits = F.interpolate(logits, size=true_mask.shape, mode='bilinear', align_corners=False)
        pred_mask = logits.argmax(dim=1).squeeze().cpu().numpy()

    score = compute_official_iou(pred_mask, true_mask)
    results.append((img_name, original_img, true_mask, pred_mask, score))
    
    for c in range(10):
        i, u = ((pred_mask==c)&(true_mask==c)).sum(), ((pred_mask==c)|(true_mask==c)).sum()
        if u > 0: class_ious[c].append(i/u)
    
    all_preds.append(pred_mask.flatten()[::100])
    all_targets.append(true_mask.flatten()[::100])

# ==========================================
# 6. GENERATE REPORTS
# ==========================================
results.sort(key=lambda x: x[4])
avg_iou = sum([x[4] for x in results])/len(results) if results else 0.0

with open(f"{OUTPUT_DIR}/evaluation_metrics.txt", "w") as f:
    f.write(f"DINOv2 (ViT-L/14) OFFICIAL MEAN IoU: {avg_iou:.4f}\n\n")
    class_avgs = [sum(class_ious[i])/len(class_ious[i]) if class_ious[i] else 0.0 for i in range(10)]
    for i, name in enumerate(CLASS_NAMES):
        f.write(f"{name:<15}: {class_avgs[i]:.4f}\n")

plt.figure(figsize=(10,6))
plt.bar(CLASS_NAMES, class_avgs, color=[c/255 for c in CLASS_COLORS])
plt.xticks(rotation=45); plt.title(f'DINOv2 Per-Class IoU (Mean: {avg_iou:.2f})'); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/per_class_metrics.png")

cm = confusion_matrix(np.concatenate(all_targets), np.concatenate(all_preds), labels=range(10))
plt.figure(figsize=(10,8))
sns.heatmap(cm.astype('float')/cm.sum(axis=1)[:,np.newaxis], cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")

def save_vis(cases, folder):
    for n, img, true, pred, s in cases:
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1); plt.imshow(img); plt.axis('off')
        plt.subplot(1,3,2); plt.imshow(decode_mask(true)); plt.axis('off')
        plt.subplot(1,3,3); plt.imshow(decode_mask(pred)); plt.title(f"IoU: {s:.2f}"); plt.axis('off')
        plt.savefig(f"{OUTPUT_DIR}/{folder}/{n}_vis.png"); plt.close()
        overlay = cv2.addWeighted(img, 1, decode_mask(pred), 0.5, 0)
        cv2.imwrite(f"{OUTPUT_DIR}/overlays/{n}_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

save_vis(results[:5], "failure_cases")
save_vis(results[-5:], "visualizations")

print(f"🏆 DINOv2 Complete! Mean IoU: {avg_iou:.4f}")