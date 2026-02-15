---
title: Semantic Segmentation Ensemble
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

🚜 Semantic Segmentation Ensemble for Off-Road Navigation

A high-precision semantic segmentation pipeline designed for autonomous off-road navigation. This project uses an ensemble of 7 deep learning models (including SegFormer, DINOv2, and U-Net++) to accurately classify complex terrain features like logs, puddles, dry grass, and obstacles.
🎥 Demo Video

    [INSERT YOUR YOUTUBE VIDEO LINK HERE]

    Watch the full walkthrough of the Inference Dashboard and our Requestly API workflow.

🧠 Project Architecture

Off-road environments are chaotic. A single model often fails to distinguish between safe terrain (e.g., dry grass) and obstacles (e.g., logs). Our solution uses a Multi-Phase Ensemble Strategy:
The Ensemble Engine

We employ a Weighted Soft-Voting Mechanism that combines predictions from 7 distinct architectures:

    Phase 1: DeepLabV3+ (ResNet101 Backbone) - Baseline Stability

    Phase 2: DINOv2 (ViT-B/14) - Foundation Model Feature Extraction

    Phase 3: EfficientNet-B4 - Lightweight Accuracy

    Phase 4: SegFormer - Transformer-based Global Context

    Phase 5: U-Net++ - Fine-grained Boundary Detection

    Phase 6: Feature Pyramid Networks (FPN) - Multi-scale Object Detection

Key Features

    Test-Time Augmentation (TTA): The model performs horizontal flips during inference to ensure robustness.

    Class Boosting: Dynamic weight adjustment for rare but critical classes (e.g., heavily weighting "Logs" to prevent collision).

    Production API: A robust Flask backend serving the ensemble via a RESTful API.

    Dockerized Deployment: Fully containerized for deployment on Hugging Face Spaces or any cloud provider.

🛠️ Installation & Setup
Prerequisites

    Git & Git LFS (Large File Storage is critical for downloading model weights).

    Docker (Recommended for easiest setup).

    Python 3.9+ (If running locally).

Option A: Running with Docker (Recommended)

This is the fastest way to get started without worrying about dependencies.

    Clone the Repository:
    Bash

    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME

    Build the Image:
    Bash

    docker build -t semantic-segmentation-app .

    Run the Container:
    Bash

    docker run -p 7860:7860 semantic-segmentation-app

    Access the App:
    Open your browser and go to http://localhost:7860

Option B: Local Python Setup

    Install Git LFS:
    Bash

    git lfs install
    git lfs pull  # Downloads the large .pth model files

    Create a Virtual Environment:
    Bash

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    Install Dependencies:
    Note: We use opencv-python-headless for server environments.
    Bash

    pip install -r web_interface/requirements.txt

    Run the Flask App:
    Bash

    python web_interface/app.py

    Access the app at http://127.0.0.1:7860.

📡 API Documentation

This project exposes a REST API for integration with autonomous vehicle control systems.
POST /predict

Uploads an image and returns the segmentation mask and confidence score.

    URL: /predict

    Method: POST

    Content-Type: multipart/form-data

    Body:

        file: The image file (jpg/png)

    Response:
    JSON

    {
      "message": "Segmentation Successful",
      "original_url": "/static/uploads/1234_input.png",
      "mask_url": "/static/predictions/mask_1234.png",
      "score": "0.7452"
    }

🧪 Tested with Requestly

We use Requestly to validate our API endpoints across environments.

    Environment Variables: Used to toggle between Local Dev and Production without code changes.

    Automated Testing: Custom Post-Response scripts validate that the ensemble returns a valid mask_url and status: 200.

📂 Project Structure
Plaintext

SEMANTIC-SEGMENTATION/
├── Models/
│   ├── Ensemble/
│   │   ├── single_inference.py  # Core Inference Engine
│   │   ├── model_utils.py       # Voting & TTA Logic
│   │   └── Result/              # Metrics & Charts
│   ├── dino/                    # DINOv2 Model & Weights
│   ├── phase1/                  # DeepLabV3+ Weights
│   └── ... (other phases)
├── web_interface/
│   ├── app.py                   # Flask Server Entrypoint
│   ├── static/                  # CSS, JS, Uploads
│   └── templates/               # HTML (Inference & Analysis Pages)
├── Dockerfile                   # Production Docker Configuration
└── requirements.txt             # Python Dependencies

🤝 Acknowledgements

    Requestly: For providing the API client used to debug and test our inference endpoints.

    Hugging Face: For hosting the model weights and the live demo space.

    Timm Library: For the efficient model backbones.