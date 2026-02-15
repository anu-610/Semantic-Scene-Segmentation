import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. IMPORT UTILS (Simple Import)
# ==========================================
try:
    import model_utils
    print("✅ model_utils loaded successfully.")
except ImportError:
    print("❌ CRITICAL ERROR: Could not find 'model_utils.py' in the same folder.")
    sys.exit(1)

# ==========================================
# 2. CONFIGURATION
# ==========================================
# Detect current folder for output
try:
    current_script_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_script_path = os.getcwd()

TEST_IMG_DIR = "/home/dev/Documents/Hackathon/Krack Hack/ensemble/Offroad_Segmentation_testImages/Color_Images"
TEST_MASK_DIR = "/home/dev/Documents/Hackathon/Krack Hack/ensemble/Offroad_Segmentation_testImages/Segmentation"
OUTPUT_DIR = os.path.join(current_script_path, "Result")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- MODEL PATHS ---
dino = model_utils.get_model_path("best_model_dino.pth", "/kaggle/input/models/anurajktk/dinorishi/pytorch/default/1/best_model (2).pth")
krackhack1 = model_utils.get_model_path("best_model.pth", "/kaggle/input/models/anurajktk/krackhackp2v1/pytorch/default/1/best_model_phase2.pth")
phase1 = model_utils.get_model_path("best_model_phase1_remastered.pth", "/kaggle/input/models/anurajktk/phase1/pytorch/default/1/best_model_phase1_remastered.pth")
phase3 = model_utils.get_model_path("best_model_phase3_effb4.pth", "/kaggle/input/models/anurajktk/phase3/pytorch/default/1/best_model_phase3_effb4.pth")
phase4 = model_utils.get_model_path("best_model_phase4_segformer.pth", "/kaggle/input/models/anurajktk/phase4/pytorch/default/1/best_model_phase4_segformer.pth")
phase5 = model_utils.get_model_path("best_model_phase5_unetplusplus.pth", "/kaggle/input/models/anurajktk/phase5/pytorch/default/1/best_model_phase5_unetplusplus.pth")
phase6 = model_utils.get_model_path("best_model_phase6_fpn.pth", "/kaggle/input/models/anurajktk/phase6/pytorch/default/1/best_model_phase6_fpn.pth")

# --- MODEL CONFIGURATION ---
MODEL_CONFIG = [
    ("DINOv2", dino, "DINO", "dinov2_vitl14", 4.0),
    ("Phase1", krackhack1, "DeepLabV3+", "resnet101", 2.0),
    ("Phase1_Remastered", phase1, "DeepLabV3+", "resnet101", 1.0),
    ("Phase3", phase3, "DeepLabV3+", "timm-efficientnet-b4", 1.0),
    ("Phase5", phase5, "UnetPlusPlus", "resnet34", 1.0),
    ("Phase6", phase6, "FPN", "resnext50_32x4d", 1.0),
    ("Phase4", phase4, "DeepLabV3+", "mit_b2", 0.5), 
]

# --- THE BOOST MAP ---
CLASS_BOOSTS = [0.0, 0.0, 1.5, 0.0, 1.2, 0.0, 3.0, 1.5, -0.5, 0.0]
DINO_IMG_SIZE = (616, 616)
VALUE_MAP = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}
CLASS_NAMES = ['Background', 'Trees', 'Lush Bush', 'Dry Grass', 'Dry Bush', 'Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']
CLASS_COLORS = np.array([
    [0, 0, 0], [34, 139, 34], [0, 255, 0], [210, 180, 140], [139, 90, 43],
    [128, 128, 0], [139, 69, 19], [128, 128, 128], [160, 82, 45], [135, 206, 235]
], dtype=np.uint8)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/failure_cases", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/overlays", exist_ok=True)

# ==========================================
# 3. DEFINITIONS & UTILS
# ==========================================
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
        h_new = int(np.sqrt(N * self.H / self.W))
        w_new = int(np.sqrt(N * self.W / self.H))
        x = x.reshape(B, h_new, w_new, C).permute(0, 3, 1, 2)
        if h_new != self.H or w_new != self.W:
             x = F.interpolate(x, size=(self.H, self.W), mode='bilinear', align_corners=False)
        return self.classifier(self.block(self.stem(x)))

def compute_official_iou(pred_mask, true_mask, num_classes=10):
    iou_per_class = []
    for class_id in range(num_classes):
        pred_inds = (pred_mask == class_id)
        target_inds = (true_mask == class_id)
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        if union > 0:
            iou_per_class.append(intersection / union)
    return sum(iou_per_class) / len(iou_per_class) if len(iou_per_class) > 0 else 0.0

def decode_mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for id, color in enumerate(CLASS_COLORS):
        rgb[mask == id] = color
    return rgb

def create_overlay(image, mask, alpha=0.5):
    color_mask = decode_mask_to_rgb(mask)
    return cv2.addWeighted(image, 1, color_mask, alpha, 0)

# ==========================================
# 4. LOAD MODELS
# ==========================================
loaded_models = []
print("⏳ Loading Models...")

for name, path, arch, encoder, weight in MODEL_CONFIG:
    if not os.path.exists(path):
        print(f"⚠️ Warning: {path} not found. Skipping.")
        continue
    
    print(f"   Loading {name} (Weight: {weight})...")
    
    if arch == "DINO":
        backbone = torch.hub.load('facebookresearch/dinov2', encoder).to(DEVICE).eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, *DINO_IMG_SIZE).to(DEVICE)
            out = backbone.forward_features(dummy)["x_norm_patchtokens"]
            embed_dim = out.shape[2]
        
        head = SegmentationHeadConvNeXt(embed_dim, 10, DINO_IMG_SIZE[1]//14, DINO_IMG_SIZE[0]//14).to(DEVICE)
        head.load_state_dict(torch.load(path, map_location=DEVICE))
        head.eval()
        loaded_models.append({'type': 'dino', 'backbone': backbone, 'head': head, 'weight': weight})
        
    else: 
        if arch == "DeepLabV3+":
            model = smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=None, classes=10, activation=None)
        elif arch == "UnetPlusPlus":
            model = smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=None, classes=10, activation=None)
        elif arch == "FPN":
            model = smp.FPN(encoder_name=encoder, encoder_weights=None, classes=10, activation=None)
            
        state_dict = torch.load(path, map_location=DEVICE)
        if 'module.' in list(state_dict.keys())[0]:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        loaded_models.append({'type': 'smp', 'model': model, 'weight': weight})

# ==========================================
# 5. INFERENCE WITH BOOST, TTA & REPORTING
# ==========================================
smp_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
dino_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
boost_tensor = torch.tensor(CLASS_BOOSTS).to(DEVICE).view(1, 10, 1, 1)

def get_logits(model_info, image_cv2, image_pil, target_shape):
    if model_info['type'] == 'smp':
        x = smp_transform(image=image_cv2)["image"].unsqueeze(0).to(DEVICE)
        logits = model_info['model'](x)
    else:
        x = TF.resize(image_pil, DINO_IMG_SIZE, Image.BILINEAR)
        x = TF.to_tensor(x)
        x = dino_normalize(x).unsqueeze(0).to(DEVICE)
        feats = model_info['backbone'].forward_features(x)["x_norm_patchtokens"]
        logits = model_info['head'](feats)
    prob_orig = F.interpolate(logits, size=target_shape, mode='bilinear', align_corners=False)
    
    if model_info['type'] == 'smp':
        image_flipped = cv2.flip(image_cv2, 1)
        x_flip = smp_transform(image=image_flipped)["image"].unsqueeze(0).to(DEVICE)
        logits_flip = model_info['model'](x_flip)
    else:
        image_flipped = TF.hflip(image_pil)
        x = TF.resize(image_flipped, DINO_IMG_SIZE, Image.BILINEAR)
        x = TF.to_tensor(x)
        x = dino_normalize(x).unsqueeze(0).to(DEVICE)
        feats = model_info['backbone'].forward_features(x)["x_norm_patchtokens"]
        logits_flip = model_info['head'](feats)
    prob_flip = F.interpolate(logits_flip, size=target_shape, mode='bilinear', align_corners=False)
    prob_flip = torch.flip(prob_flip, dims=[3]) 
    
    return (prob_orig + prob_flip) / 2.0

results = []
all_preds = []
all_targets = []
class_iou_totals = {i: [] for i in range(10)}

if not os.path.exists(TEST_IMG_DIR):
    print(f"❌ TEST IMAGE DIR NOT FOUND: {TEST_IMG_DIR}")
    print("Please update 'TEST_IMG_DIR' in the script to your local image folder.")
    sys.exit(1)

test_files = sorted(os.listdir(TEST_IMG_DIR))
print(f"🚀 Running BOOSTED ENSEMBLE on {len(test_files)} images...")

for img_name in tqdm(test_files):
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    mask_path = os.path.join(TEST_MASK_DIR, img_name)
    if not os.path.exists(mask_path): continue

    img_cv2 = cv2.imread(img_path); img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.open(img_path).convert("RGB")
    raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    true_mask = np.zeros_like(raw_mask, dtype=np.uint8)
    for k, v in VALUE_MAP.items(): true_mask[raw_mask == k] = v
    
    total_weighted_logits = None
    with torch.no_grad():
        for model_info in loaded_models:
            logits = get_logits(model_info, img_cv2, img_pil, true_mask.shape)
            w = model_info['weight']
            if total_weighted_logits is None:
                total_weighted_logits = logits * w
            else:
                total_weighted_logits += (logits * w)
    
    total_weighted_logits += boost_tensor
    pred_mask = total_weighted_logits.argmax(dim=1).squeeze().cpu().numpy()
    
    score = compute_official_iou(pred_mask, true_mask)
    results.append((img_name, img_cv2, true_mask, pred_mask, score))
    
    for cls_id in range(10):
        intersection = ((pred_mask == cls_id) & (true_mask == cls_id)).sum()
        union = ((pred_mask == cls_id) | (true_mask == cls_id)).sum()
        if union > 0: class_iou_totals[cls_id].append(intersection / union)

    all_preds.append(pred_mask.flatten()[::100]) 
    all_targets.append(true_mask.flatten()[::100])

# ==========================================
# 6. REPORT GENERATION
# ==========================================
results.sort(key=lambda x: x[4]) 
worst_cases = results[:5]      
best_cases = results[-5:]      

avg_iou = sum([x[4] for x in results]) / len(results) if results else 0
with open(f"{OUTPUT_DIR}/evaluation_metrics.txt", "w") as f:
    f.write(f"BOOSTED ENSEMBLE SCORE: {avg_iou:.4f}\n\nPer-Class Breakdown:\n")
    class_avgs = []
    for i in range(10):
        if len(class_iou_totals[i]) > 0: c_avg = sum(class_iou_totals[i])/len(class_iou_totals[i])
        else: c_avg = 0.0
        class_avgs.append(c_avg)
        f.write(f"{CLASS_NAMES[i]:<15}: {c_avg:.4f}\n")
    f.write("\nWorst 5 Images:\n")
    for item in worst_cases: f.write(f"{item[0]}: {item[4]:.4f}\n")
print(f"✅ Text Report Saved. Score: {avg_iou:.4f}")

plt.figure(figsize=(10, 6))
plt.bar(CLASS_NAMES, class_avgs, color=[c/255 for c in CLASS_COLORS])
plt.xticks(rotation=45, ha='right')
plt.title(f'Boosted Ensemble Per-Class IoU (Mean: {avg_iou:.2f})')
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/per_class_metrics.png")
print("✅ Bar Chart Saved.")

print("📊 Generating Confusion Matrix...")
cm = confusion_matrix(np.concatenate(all_targets), np.concatenate(all_preds), labels=range(10))
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=False, cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Boosted Ensemble Confusion Matrix')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
print("✅ Confusion Matrix Saved.")

def save_visuals(case_list, folder_name):
    for img_name, img, true, pred, score in case_list:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.imshow(img); plt.title("Original"); plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(decode_mask_to_rgb(true)); plt.title("Ground Truth"); plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(decode_mask_to_rgb(pred)); plt.title(f"Pred (IoU: {score:.2f})"); plt.axis('off')
        plt.savefig(f"{OUTPUT_DIR}/{folder_name}/{img_name}_comparison.png")
        plt.close()
        overlay = create_overlay(img, pred)
        cv2.imwrite(f"{OUTPUT_DIR}/overlays/{img_name}_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

save_visuals(worst_cases, "failure_cases")
save_visuals(best_cases, "visualizations")

print(f"🏆 ALL DONE! Full Report in {OUTPUT_DIR}")