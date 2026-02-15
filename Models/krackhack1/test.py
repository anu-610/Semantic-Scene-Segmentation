import sys
import os

# ==========================================
# 1. IMPORT UTILS
# ==========================================
try:
    import model_utils
    print("✅ model_utils loaded successfully.")
except ImportError:
    print("❌ CRITICAL ERROR: Could not find 'model_utils.py' in the same folder.")
    sys.exit(1)

# ==========================================
# 2. STANDARD IMPORTS
# ==========================================
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

# ==========================================
# 3. CONFIGURATION
# ==========================================
try:
    current_script_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_script_path = os.getcwd()

TEST_IMG_DIR = "/home/dev/Documents/Hackathon/Krack Hack/ensemble/Offroad_Segmentation_testImages/Color_Images"
TEST_MASK_DIR = "/home/dev/Documents/Hackathon/Krack Hack/ensemble/Offroad_Segmentation_testImages/Segmentation"
OUTPUT_DIR = os.path.join(current_script_path, "Submission_Results_Phase1_TTA")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- MODEL PATH (FAILSAFE) ---
MODEL_PATH = model_utils.get_model_path(
    "best_model_phase1_remastered.pth", 
    "/kaggle/input/models/anurajktk/krackhackmodel1/pytorch/default/2/best_model.pth"
)

# 16-bit Mapping
VALUE_MAP = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
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
# 4. UTILS
# ==========================================
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
# 5. MODEL SETUP (The PyTorch 2.6 Fix)
# ==========================================
print(f"⏳ Loading Model from {MODEL_PATH}...")

model = smp.DeepLabV3Plus(encoder_name="resnet101", classes=10, activation=None)

# 🔥 FIX: Added weights_only=False for PyTorch 2.6 compatibility
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

if 'module.' in list(state_dict.keys())[0]:
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ==========================================
# 6. INFERENCE LOOP
# ==========================================
print("🚀 Running Inference...")
results = []
all_preds, all_targets = [], []
class_iou_totals = {i: [] for i in range(10)}

test_files = sorted(os.listdir(TEST_IMG_DIR))

for img_name in tqdm(test_files):
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    mask_path = os.path.join(TEST_MASK_DIR, img_name)
    if not os.path.exists(mask_path): continue

    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    true_mask = np.zeros_like(raw_mask, dtype=np.uint8)
    for k, v in VALUE_MAP.items(): true_mask[raw_mask == k] = v

    img_tensor = transform(image=original_img)["image"].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits_orig = model(img_tensor)
        logits_orig = F.interpolate(logits_orig, size=true_mask.shape, mode='bilinear', align_corners=False)
        probs_orig = F.softmax(logits_orig, dim=1)

        img_flip = torch.flip(img_tensor, dims=[3])
        logits_flip = model(img_flip)
        logits_flip = F.interpolate(logits_flip, size=true_mask.shape, mode='bilinear', align_corners=False)
        probs_flip = torch.flip(F.softmax(logits_flip, dim=1), dims=[3])

        pred_mask = ((probs_orig + probs_flip) / 2.0).argmax(dim=1).squeeze().cpu().numpy()

    score = compute_official_iou(pred_mask, true_mask)
    results.append((img_name, original_img, true_mask, pred_mask, score))
    
    for cls_id in range(10):
        inter = ((pred_mask == cls_id) & (true_mask == cls_id)).sum()
        union = ((pred_mask == cls_id) | (true_mask == cls_id)).sum()
        if union > 0: class_iou_totals[cls_id].append(inter / union)
            
    all_preds.append(pred_mask.flatten()[::100]) 
    all_targets.append(true_mask.flatten()[::100])

# ==========================================
# 7. SAVING REPORTS
# ==========================================
results.sort(key=lambda x: x[4]) 
avg_iou = sum([x[4] for x in results]) / len(results)

# Safety check for folder before writing
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(f"{OUTPUT_DIR}/evaluation_metrics.txt", "w") as f:
    f.write(f"OFFICIAL MEAN IoU: {avg_iou:.4f}\n\nPer-Class Breakdown:\n")
    class_avgs = [sum(class_iou_totals[i])/len(class_iou_totals[i]) if class_iou_totals[i] else 0.0 for i in range(10)]
    for i, name in enumerate(CLASS_NAMES):
        f.write(f"{name:<15}: {class_avgs[i]:.4f}\n")

plt.figure(figsize=(10, 6))
plt.bar(CLASS_NAMES, class_avgs, color=[c/255 for c in CLASS_COLORS])
plt.xticks(rotation=45, ha='right'); plt.ylim(0, 1.0); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/per_class_metrics.png")

save_visuals = lambda case_list, folder: [ (plt.figure(figsize=(15,5)), plt.subplot(1,3,1), plt.imshow(img), plt.subplot(1,3,2), plt.imshow(decode_mask_to_rgb(true)), plt.subplot(1,3,3), plt.imshow(decode_mask_to_rgb(pred)), plt.title(f"IoU: {s:.2f}"), plt.savefig(f"{OUTPUT_DIR}/{folder}/{n}_comparison.png"), plt.close(), cv2.imwrite(f"{OUTPUT_DIR}/overlays/{n}_overlay.png", cv2.cvtColor(create_overlay(img, pred), cv2.COLOR_RGB2BGR))) for n, img, true, pred, s in case_list ]

save_visuals(results[:5], "failure_cases")
save_visuals(results[-5:], "visualizations")

print(f"🏆 ALL DONE! MEAN IoU: {avg_iou:.4f}")