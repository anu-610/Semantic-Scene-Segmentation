import os
import requests
import sys

# ==========================================
# CONFIGURATION
# ==========================================
# Models are stored in the PARENT folder (Models/) 
# so they are shared across all phases.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

# ⚠️ UPDATE THESE WITH YOUR REAL GITHUB LINKS
MODEL_URLS = {
    "best_model_dino.pth": "https://github.com/anu-610/Semantic-Scene-Segmentation/raw/refs/heads/main/Models/DinoRishi.pth",
    "best_model": "https://github.com/anu-610/Semantic-Scene-Segmentation/raw/refs/heads/main/Models/best_model.pth",
    "best_model_phase1_remastered.pth": "https://github.com/anu-610/Semantic-Scene-Segmentation/raw/refs/heads/main/Models/best_model_phase1_remastered.pth",
    "best_model_phase3_effb4.pth": "https://github.com/anu-610/Semantic-Scene-Segmentation/raw/refs/heads/main/Models/best_model_phase3_effb4.pth",
    "best_model_phase4_segformer.pth": "https://github.com/anu-610/Semantic-Scene-Segmentation/raw/refs/heads/main/Models/best_model_phase4_segformer.pth",
    "best_model_phase5_unetplusplus.pth": "https://github.com/anu-610/Semantic-Scene-Segmentation/raw/refs/heads/main/Models/best_model_phase5_unetplusplus.pth",
    "best_model_phase6_fpn.pth": "https://github.com/anu-610/Semantic-Scene-Segmentation/raw/refs/heads/main/Models/best_model_phase6_fpn.pth",
}

def get_model_path(filename, kaggle_fallback_path=None):
    """
    Robust Path Resolver:
    1. Checks Kaggle Path (Fastest)
    2. Checks Local 'Models/' Folder (Parent of this script)
    3. Downloads from GitHub if missing
    """
    
    # 1. CHECK KAGGLE (Priority 1)
    if kaggle_fallback_path and os.path.exists(kaggle_fallback_path):
        print(f"✅ Found on Kaggle: {kaggle_fallback_path}")
        return kaggle_fallback_path

    # 2. CHECK LOCAL (Priority 2)
    if not os.path.exists(LOCAL_MODEL_DIR):
        os.makedirs(LOCAL_MODEL_DIR)
        
    target_path = os.path.join(LOCAL_MODEL_DIR, filename)

    if os.path.exists(target_path):
        print(f"✅ Found locally: {target_path}")
        return target_path

    # 3. DOWNLOAD (Failsafe)
    print(f"⚠️ Model '{filename}' missing in {LOCAL_MODEL_DIR}. Downloading from GitHub...")
    
    url = MODEL_URLS.get(filename)
    if not url:
        print(f"❌ ERROR: URL for '{filename}' not found in MODEL_URLS dict.")
        sys.exit(1)
        
    try:
        print(f"⬇️ Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk: f.write(chunk)
                
        print(f"✅ Download Complete: {target_path}")
        return target_path
        
    except Exception as e:
        print(f"❌ DOWNLOAD FAILED: {e}")
        if os.path.exists(target_path): os.remove(target_path)
        sys.exit(1)