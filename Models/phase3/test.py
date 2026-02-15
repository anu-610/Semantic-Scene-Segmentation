import sys
import os

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

# ==========================================
# 3. CONFIGURATION
# ==========================================
try:
    current_script_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_script_path = os.getcwd()

TEST_IMG_DIR = "/kaggle/input/datasets/devil610/testdata-krackhack/Offroad_Segmentation_testImages/Color_Images"
TEST_MASK_DIR = "/kaggle/input/datasets/devil610/testdata-krackhack/Offroad_Segmentation_testImages/Segmentation"
OUTPUT_DIR = os.path.join(current_script_path, "TestResult")

# --- MODEL PATH (FAILSAFE) ---
# Resolves to shared Models/ folder or downloads if missing
MODEL_PATH = model_utils.get_model_path(
    "best_model_phase3_effb4.pth", 
    "/kaggle/input/models/devil610/phase3/pytorch/default/1/best_model_phase3_effb4.pth"
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 16-bit Mapping & Classes
VALUE_MAP = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}
CLASS_NAMES = ['Background', 'Trees', 'Lush Bush', 'Dry Grass', 'Dry Bush', 'Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']
CLASS_COLORS = np.array([[0,0,0], [34,139,34], [0,255,0], [210,180,140], [139,90,43], [128,128,0], [139,69,19], [128,128,128], [160,82,45], [135,206,235]], dtype=np.uint8)

# Create Output Folders
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/failure_cases", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/overlays", exist_ok=True)

# ==========================================
# 4. UTILS
# ==========================================
def compute_official_iou(pred, true, num_classes=10):
    iou_list = []
    for c in range(num_classes):
        i = ((pred == c) & (true == c)).sum()
        u = ((pred == c) | (true == c)).sum()
        if u > 0: iou_list.append(i/u)
    return sum(iou_list)/len(iou_list) if len(iou_list) > 0 else 0.0

def decode_mask(mask):
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, c in enumerate(CLASS_COLORS): rgb[mask == i] = c
    return rgb


def sliding_window_inference(model, image, window_size=512, stride=384):
    h, w, _ = image.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    
    probs = np.zeros((image.shape[0], image.shape[1], 10))
    counts = np.zeros((image.shape[0], image.shape[1], 10))
    transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
    
    for y in range(0, image.shape[0]-window_size+1, stride):
        for x in range(0, image.shape[1]-window_size+1, stride):
            patch = image[y:y+window_size, x:x+window_size]
            tensor = transform(image=patch)['image'].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy().transpose(1,2,0)
            probs[y:y+window_size, x:x+window_size] += out
            counts[y:y+window_size, x:x+window_size] += 1
    return np.argmax(probs/counts, axis=2)[:h, :w].astype(np.uint8)

# ==========================================
# 5. LOAD MODEL (PyTorch 2.6 Fix Included)
# ==========================================
print(f"⏳ Loading Phase 3 Model: {MODEL_PATH}")
model = smp.DeepLabV3Plus(encoder_name="timm-efficientnet-b4", classes=10, activation=None)

# 🔥 weights_only=False is required for PyTorch 2.6+ to load legacy/custom models
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

if 'module.' in list(state_dict.keys())[0]:
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
model.load_state_dict(state_dict)
model.to(DEVICE); model.eval()

# ==========================================
# 6. RUN INFERENCE
# ==========================================
results, all_preds, all_targets, class_ious = [], [], [], {i: [] for i in range(10)}
print("🚀 Phase 3 Inference Started...")

test_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])

for img_name in tqdm(test_files):
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    mask_path = os.path.join(TEST_MASK_DIR, img_name)
    if not os.path.exists(mask_path): continue
    
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    true_mask = np.zeros_like(raw_mask, dtype=np.uint8)
    for k, v in VALUE_MAP.items(): true_mask[raw_mask == k] = v
    
    pred_mask = sliding_window_inference(model, img)
    score = compute_official_iou(pred_mask, true_mask)
    
    results.append((img_name, img, true_mask, pred_mask, score))
    all_preds.append(pred_mask.flatten()[::100]); all_targets.append(true_mask.flatten()[::100])
    for c in range(10):
        i, u = ((pred_mask==c)&(true_mask==c)).sum(), ((pred_mask==c)|(true_mask==c)).sum()
        if u > 0: class_ious[c].append(i/u)

# ==========================================
# 7. GENERATE OUTPUTS
# ==========================================
results.sort(key=lambda x: x[4])
avg_iou = sum([x[4] for x in results])/len(results) if results else 0.0

# 1. Metrics Text
with open(f"{OUTPUT_DIR}/evaluation_metrics.txt", "w") as f:
    f.write(f"PHASE 3 OFFICIAL MEAN IoU: {avg_iou:.4f}\n\nPer-Class Breakdown:\n")
    class_avgs = []
    for c in range(10):
        avg = sum(class_ious[c])/len(class_ious[c]) if class_ious[c] else 0.0
        class_avgs.append(avg)
        f.write(f"{CLASS_NAMES[c]:<15}: {avg:.4f}\n")
    f.write("\nWorst 5 Failure Cases:\n")
    for x in results[:5]: f.write(f"{x[0]}: {x[4]:.4f}\n")

# 2. Bar Chart
plt.figure(figsize=(10,6))
plt.bar(CLASS_NAMES, class_avgs, color=[c/255 for c in CLASS_COLORS])
plt.title(f'Phase 3 Per-Class IoU (Mean: {avg_iou:.2f})'); plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/per_class_iou_chart.png")

# 3. Confusion Matrix

cm = confusion_matrix(np.concatenate(all_targets), np.concatenate(all_preds), labels=range(10))
plt.figure(figsize=(10,8))
sns.heatmap(cm.astype('float')/cm.sum(axis=1)[:,np.newaxis], cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Phase 3 Confusion Matrix"); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")

# 4. Visualizations
def save_vis(cases, folder):
    for name, img, true, pred, score in cases:
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1); plt.imshow(img); plt.title("Original"); plt.axis('off')
        plt.subplot(1,3,2); plt.imshow(decode_mask(true)); plt.title("Ground Truth"); plt.axis('off')
        plt.subplot(1,3,3); plt.imshow(decode_mask(pred)); plt.title(f"Pred (IoU: {score:.2f})"); plt.axis('off')
        plt.savefig(f"{OUTPUT_DIR}/{folder}/{name}_vis.png"); plt.close()
        overlay = cv2.addWeighted(img, 1, decode_mask(pred), 0.5, 0)
        cv2.imwrite(f"{OUTPUT_DIR}/overlays/{name}_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

save_vis(results[:5], "failure_cases")
save_vis(results[-5:], "visualizations")

print(f"✅ Phase 3 Complete! Check folder: {OUTPUT_DIR}")