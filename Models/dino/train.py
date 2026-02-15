import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import RandomErasing

# ============================================================================
# 1. Configuration
# ============================================================================
CONFIG = {
    "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "BATCH_SIZE": 8,          
    "NUM_EPOCHS": 20,         
    "LEARNING_RATE": 2e-4,    
    "IMG_SIZE": (616, 616),   # DINOv2 patch size compatible (multiple of 14)
    "NUM_CLASSES": 10,
    "BACKBONE": "dinov2_vitl14",
    # Paths (Update these for your local environment)
    "TRAIN_ROOT": '/home/dev/Documents/Hackathon/Krack Hack/ensemble/Offroad_Segmentation_Training_Dataset/train',
    "VAL_ROOT": '/home/dev/Documents/Hackathon/Krack Hack/ensemble/Offroad_Segmentation_Training_Dataset/val',
    "OUTPUT_DIR": './improved_experiment'
}

# ============================================================================
# 2. Dataset & Augmentation
# ============================================================================
VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in VALUE_MAP.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

class JointTransform:
    def __init__(self, size, mode='train'):
        self.size = size
        self.mode = mode
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.eraser = RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random')

    def __call__(self, image, mask):
        # Initial Resize to avoid processing huge images
        if self.mode == 'train':
            # Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random Resized Crop
            i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.8, 1.0), ratio=(0.9, 1.1))
            image = TF.resized_crop(image, i, j, h, w, self.size, Image.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w, self.size, Image.NEAREST)
            
            # Color Jitter
            image = self.color_jitter(image)
        else:
            image = TF.resize(image, self.size, Image.BILINEAR)
            mask = TF.resize(mask, self.size, Image.NEAREST)

        # To Tensor and Normalize
        image = TF.to_tensor(image)
        image = self.normalize(image)
        
        # Apply Eraser ONLY to image in training
        if self.mode == 'train':
            image = self.eraser(image)

        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask

class MaskDataset(Dataset):
    def __init__(self, root_dir, mode='train', img_size=(616, 616)):
        self.transform = JointTransform(img_size, mode)
        self.image_paths = []
        self.mask_paths = {} 
        
        if not os.path.exists(root_dir):
            print(f"⚠️ Warning: Path {root_dir} does not exist.")
            return

        for root, _, files in os.walk(root_dir):
            for f in files:
                full_path = os.path.join(root, f)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    path_parts_lower = [p.lower() for p in full_path.split(os.sep)]
                    if 'segmentation' in path_parts_lower:
                        stem = os.path.splitext(f)[0]
                        self.mask_paths[stem] = full_path
                    else:
                        self.image_paths.append(full_path)
        
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        file_stem = os.path.splitext(os.path.basename(img_path))[0]
        
        image = Image.open(img_path).convert("RGB")
        if file_stem in self.mask_paths:
            mask = Image.open(self.mask_paths[file_stem])
            mask = convert_mask(mask)
        else:
            mask = Image.new('L', image.size, 0) # Fallback

        image, mask = self.transform(image, mask)
        return image, mask

# ============================================================================
# 3. Model & Loss Functions
# ============================================================================
class DiceCELoss(nn.Module):
    def __init__(self, num_classes, weights=None, smooth=1e-6):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        loss_ce = self.ce(pred, target)
        pred_softmax = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (pred_softmax * target_one_hot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice_score = 2 * intersection / (union + self.smooth)
        loss_dice = 1 - dice_score.mean()
        return 0.5 * loss_ce + 0.5 * loss_dice

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
        h_new, w_new = self.H, self.W
        x = x.reshape(B, h_new, w_new, C).permute(0, 3, 1, 2)
        return self.classifier(self.block(self.stem(x)))

def compute_iou(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().float().item()
        union = (pred_inds | target_inds).sum().float().item()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0

# ============================================================================
# 4. Main Training Loop
# ============================================================================
def main():
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    
    # 0:Back, 1:Tree, 2:Lush, 3:DryGr, 4:DryBu, 5:Clutter, 6:Logs, 7:Rock, 8:Land, 9:Sky
    class_weights = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.5, 2.0, 5.0, 3.0, 0.5, 0.1]).to(CONFIG['DEVICE'])

    # Data Loading
    train_set = MaskDataset(CONFIG['TRAIN_ROOT'], mode='train', img_size=CONFIG['IMG_SIZE'])
    val_set = MaskDataset(CONFIG['VAL_ROOT'], mode='val', img_size=CONFIG['IMG_SIZE'])
    
    train_loader = DataLoader(train_set, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=2, pin_memory=True)

    # Model Setup
    print(f"Loading Backbone: {CONFIG['BACKBONE']}...")
    backbone = torch.hub.load('facebookresearch/dinov2', CONFIG['BACKBONE']).to(CONFIG['DEVICE']).eval()
    
    # Determine embedding dimension dynamically
    with torch.no_grad():
        dummy_feat = backbone.forward_features(torch.randn(1, 3, *CONFIG['IMG_SIZE']).to(CONFIG['DEVICE']))["x_norm_patchtokens"]
        embed_dim = dummy_feat.shape[-1]
        
    head = SegmentationHeadConvNeXt(embed_dim, CONFIG['NUM_CLASSES'], CONFIG['IMG_SIZE'][1]//14, CONFIG['IMG_SIZE'][0]//14).to(CONFIG['DEVICE'])
    
    optimizer = optim.AdamW(head.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['NUM_EPOCHS'])
    criterion = DiceCELoss(CONFIG['NUM_CLASSES'], weights=class_weights)
    scaler = torch.cuda.amp.GradScaler() 

    history = {'train_loss': [], 'train_iou': [], 'val_iou': []}
    best_iou = 0.0

    for epoch in range(CONFIG['NUM_EPOCHS']):
        # --- TRAINING PHASE ---
        head.train()
        epoch_train_loss, epoch_train_iou = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']} [Train]")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(CONFIG['DEVICE']), masks.to(CONFIG['DEVICE'])
            
            with torch.no_grad():
                features = backbone.forward_features(imgs)["x_norm_patchtokens"]
            
            with torch.cuda.amp.autocast():
                preds = head(features)
                preds = F.interpolate(preds, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(preds, masks)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_train_loss += loss.item()
            epoch_train_iou += compute_iou(preds, masks, CONFIG['NUM_CLASSES'])
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        scheduler.step()

        # --- VALIDATION PHASE ---
        head.eval()
        epoch_val_iou = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(CONFIG['DEVICE']), masks.to(CONFIG['DEVICE'])
                features = backbone.forward_features(imgs)["x_norm_patchtokens"]
                preds = head(features)
                preds = F.interpolate(preds, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                epoch_val_iou += compute_iou(preds, masks, CONFIG['NUM_CLASSES'])

        # Metrics logging
        metrics = {
            'train_loss': epoch_train_loss / len(train_loader),
            'train_iou': epoch_train_iou / len(train_loader),
            'val_iou': epoch_val_iou / len(val_loader) if len(val_loader) > 0 else 0
        }
        
        for k, v in metrics.items(): history[k].append(v)
        
        print(f"\nSummary Epoch {epoch+1}: Loss: {metrics['train_loss']:.4f} | Train IoU: {metrics['train_iou']:.4f} | Val IoU: {metrics['val_iou']:.4f}")

        if metrics['val_iou'] > best_iou:
            best_iou = metrics['val_iou']
            torch.save(head.state_dict(), os.path.join(CONFIG['OUTPUT_DIR'], 'best_model.pth'))
            print(f"🌟 Saved Best Model with Val IoU: {best_iou:.4f}")

    # Plotting Results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Loss History'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('IoU History'); plt.legend()
    plt.savefig(os.path.join(CONFIG['OUTPUT_DIR'], 'training_metrics.png'))
    plt.show()

if __name__ == "__main__":
    main()