import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

# --- CONFIGURATION ---
try:
    from . import model_utils
except ImportError:
    import model_utils

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DINO_IMG_SIZE = (616, 616)

# --- CLASS DEFINITIONS ---
CLASS_NAMES = ['Background', 'Trees', 'Lush Bush', 'Dry Grass', 'Dry Bush', 'Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']
CLASS_COLORS = np.array([
    [0, 0, 0], [34, 139, 34], [0, 255, 0], [210, 180, 140], [139, 90, 43],
    [128, 128, 0], [139, 69, 19], [128, 128, 128], [160, 82, 45], [135, 206, 235]
], dtype=np.uint8)

# Mapping from your dataset IDs to 0-9 classes
VALUE_MAP = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}

CLASS_BOOSTS = [0.0, 0.0, 1.5, 0.0, 1.2, 0.0, 3.0, 1.5, -0.5, 0.0]

# --- MODEL PATHS (Ensure these point to your Models folder) ---
MODEL_CONFIG = [
    ("DINOv2", "../best_model_dino.pth", "DINO", "dinov2_vitl14", 4.0),
    ("Phase1", "../best_model.pth", "DeepLabV3+", "resnet101", 2.0),
    ("Phase1_Remastered", "../best_model_phase1_remastered.pth", "DeepLabV3+", "resnet101", 1.0),
    ("Phase3", "../best_model_phase3_effb4.pth", "DeepLabV3+", "timm-efficientnet-b4", 1.0),
    ("Phase5", "../best_model_phase5_unetplusplus.pth", "UnetPlusPlus", "resnet34", 1.0),
    ("Phase6", "../best_model_phase6_fpn.pth", "FPN", "resnext50_32x4d", 1.0),
    ("Phase4", "../best_model_phase4_segformer.pth", "DeepLabV3+", "mit_b2", 0.5), 
]

# --- HELPER CLASSES ---
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

# --- FUNCTIONS ---
import segmentation_models_pytorch as smp 

def load_ensemble_models():
    """ Load models once. """
    loaded_models = []
    print(f"⏳ Loading Ensemble Models on {DEVICE}...")
    
    # We look 2 levels up (Models/) because this script is in Models/Ensemble/
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    for name, filename, arch, encoder, weight in MODEL_CONFIG:
        path = os.path.join(base_path, filename.replace('../', '')) 
        
        if not os.path.exists(path):
            print(f"⚠️ Warning: Model {name} not found at {path}. Skipping.")
            continue
        
        print(f"   Loading {name}...")
        try:
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
                if arch == "DeepLabV3+": model = smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=None, classes=10, activation=None)
                elif arch == "UnetPlusPlus": model = smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=None, classes=10, activation=None)
                elif arch == "FPN": model = smp.FPN(encoder_name=encoder, encoder_weights=None, classes=10, activation=None)
                
                state_dict = torch.load(path, map_location=DEVICE)
                if 'module.' in list(state_dict.keys())[0]:
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                model.to(DEVICE)
                model.eval()
                loaded_models.append({'type': 'smp', 'model': model, 'weight': weight})
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
            
    print(f"✅ Loaded {len(loaded_models)} models.")
    return loaded_models

def colorize_mask(mask_indices):
    """ Converts a 2D mask of indices (0-9) to a 3D RGB image. """
    rgb_mask = np.zeros((mask_indices.shape[0], mask_indices.shape[1], 3), dtype=np.uint8)
    for id, color in enumerate(CLASS_COLORS):
        rgb_mask[mask_indices == id] = color
    return cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR) # OpenCV uses BGR

def predict_single(models, image_path, save_dir, true_mask_path=None):
    """
    Runs inference. If true_mask_path is provided, calculates IoU score and saves colorized GT.
    """
    smp_transform = A.Compose([A.Resize(512, 512), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
    dino_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    boost_tensor = torch.tensor(CLASS_BOOSTS).to(DEVICE).view(1, 10, 1, 1)

    img_cv2 = cv2.imread(image_path)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv2)
    original_size = (img_cv2.shape[0], img_cv2.shape[1]) 

    total_weighted_logits = None

    with torch.no_grad():
        for model_info in models:
            if model_info['type'] == 'smp':
                x = smp_transform(image=img_cv2)["image"].unsqueeze(0).to(DEVICE)
                logits = model_info['model'](x)
                x_flip = smp_transform(image=cv2.flip(img_cv2, 1))["image"].unsqueeze(0).to(DEVICE)
                logits_flip = model_info['model'](x_flip)
            else:
                x = TF.resize(img_pil, DINO_IMG_SIZE, Image.BILINEAR); x = TF.to_tensor(x); x = dino_normalize(x).unsqueeze(0).to(DEVICE)
                feats = model_info['backbone'].forward_features(x)["x_norm_patchtokens"]
                logits = model_info['head'](feats)
                x_f = TF.resize(TF.hflip(img_pil), DINO_IMG_SIZE, Image.BILINEAR); x_f = TF.to_tensor(x_f); x_f = dino_normalize(x_f).unsqueeze(0).to(DEVICE)
                feats_f = model_info['backbone'].forward_features(x_f)["x_norm_patchtokens"]
                logits_flip = model_info['head'](feats_f)

            prob_orig = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)
            prob_flip = F.interpolate(logits_flip, size=original_size, mode='bilinear', align_corners=False)
            prob_flip = torch.flip(prob_flip, dims=[3]) 
            avg_logits = (prob_orig + prob_flip) / 2.0
            
            w = model_info['weight']
            if total_weighted_logits is None: total_weighted_logits = avg_logits * w
            else: total_weighted_logits += (avg_logits * w)

    total_weighted_logits += boost_tensor
    pred_mask = total_weighted_logits.argmax(dim=1).squeeze().cpu().numpy()

    # Save Prediction
    filename = os.path.basename(image_path)
    pred_save_path = os.path.join(save_dir, f"mask_{filename}")
    cv2.imwrite(pred_save_path, colorize_mask(pred_mask))
    
    score = None
    gt_save_path = None

    # Handle Ground Truth
    if true_mask_path:
        raw_gt = cv2.imread(true_mask_path, cv2.IMREAD_UNCHANGED)
        if raw_gt is not None:
            # Resize GT if needed
            if raw_gt.shape != pred_mask.shape:
                raw_gt = cv2.resize(raw_gt, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Map GT values
            gt_indices = np.zeros_like(raw_gt, dtype=np.uint8)
            for k, v in VALUE_MAP.items(): gt_indices[raw_gt == k] = v
            
            # Calculate IoU
            iou_list = []
            for c in range(10):
                inter = ((pred_mask == c) & (gt_indices == c)).sum()
                union = ((pred_mask == c) | (gt_indices == c)).sum()
                if union > 0: iou_list.append(inter / union)
            
            score = sum(iou_list)/len(iou_list) if iou_list else 0.0
            
            # Save Colorized GT for display
            gt_save_path = os.path.join(save_dir, f"gt_color_{filename}")
            cv2.imwrite(gt_save_path, colorize_mask(gt_indices))

    return pred_save_path, score, gt_save_path