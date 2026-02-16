---
title: Semantic Segmentation Ensemble
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# 🚜 Semantic Segmentation Ensemble for Off-Road Navigation

A high-precision semantic segmentation pipeline designed for autonomous off-road navigation. This project uses an ensemble of 7 deep learning models (including SegFormer, DINOv2, and U-Net++) to accurately classify complex terrain features like logs, puddles, dry grass, and obstacles.<br>
## [🎥 Demo Video](https://youtu.be/4WYr3IZz1d8) <br>
*Watch the full walkthrough of the Inference Dashboard and our Requestly API workflow.

### [Live Deployment](https://huggingface.co/spaces/devil610/KrackHack)
    

## 🧠 Project Architecture<br>

Off-road environments are chaotic. A single model often fails to distinguish between safe terrain (e.g., dry grass) and obstacles (e.g., logs). Our solution uses a Multi-Phase Ensemble Strategy:<br>
The Ensemble Engine<br>

We employ a Weighted Soft-Voting Mechanism that combines predictions from 7 distinct architectures:<br>

*Phase 1: DeepLabV3+ (ResNet101 Backbone) - Baseline Stability

*Phase 2: DINOv2 (ViT-B/14) - Foundation Model Feature Extraction

*Phase 3: EfficientNet-B4 - Lightweight Accuracy

*Phase 4: SegFormer - Transformer-based Global Context

*Phase 5: U-Net++ - Fine-grained Boundary Detection

*Phase 6: Feature Pyramid Networks (FPN) - Multi-scale Object Detection

**Key Features**<br>

*Test-Time Augmentation (TTA): The model performs horizontal flips during inference to ensure robustness.

*Class Boosting: Dynamic weight adjustment for rare but critical classes (e.g., heavily weighting "Logs" to prevent collision).

*Production API: A robust Flask backend serving the ensemble via a RESTful API.

*Dockerized Deployment: Fully containerized for deployment on Hugging Face Spaces or any cloud provider.

## 🛠️ Installation & Setup<br>
**Prerequisites** <br>

*Git & Git LFS (Large File Storage is critical for downloading model weights).

*Docker (Recommended for easiest setup).

*Python 3.9+ (If running locally).

### Option A: Running with Docker (Recommended) <br>

This is the fastest way to get started without worrying about dependencies.<br>

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/anu-610/Semantic-Scene-Segmentation.git
    cd Semantic-Scene-Segmentation
    ```

2.  **Build the Image:**
    ```bash
    docker build -t semantic-segmentation-app .
    ```
  
3.  **Run the Container:**
    ```bash
    docker run -p 7860:7860 semantic-segmentation-app
    ```
4.  **Access the App:**
    Open your browser and go to [App](http://localhost:7860)

### Option B: Local Python Setup<br>

1. **Install Git LFS:**
    ```Bash
    git lfs install
    git lfs pull  # Downloads the large .pth model files
    ```
2. **Create a Virtual Environment:**
   ```Bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. **Install Dependencies:**
    *Note: We use opencv-python-headless for server environments.
   ```Bash
    pip install -r web_interface/requirements.txt
    ```
4. **Run the Flask App:**
    ```Bash
    python web_interface/app.py
    ```
5. **Access the app at [here](http://127.0.0.1:7860)**

## 📡 API Documentation<br>

This project exposes a REST API for integration with autonomous vehicle control systems.
POST /predict<br>

Uploads an image and returns the segmentation mask and confidence score.<br>

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

## 🧪 Tested with Requestly<br>

We use Requestly to validate our API endpoints across environments.<br>

***Environment Variables**: Used to toggle between Local Dev and Production without code changes.

***Automated Testing:** Custom Post-Response scripts validate that the ensemble returns a valid mask_url and status: 200.

## 📂 Project Structure<br>
SEMANTIC-SEGMENTATION/<br>
├── Models/<br>
│   ├── Ensemble/<br>
│   │   ├── single_inference.py  # Core Inference Engine<br>
│   │   ├── model_utils.py       # Voting & TTA Logic<br>
│   │   └── Result/              # Metrics & Charts<br>
│   ├── dino/                    # DINOv2 Model & Weights<br>
│   ├── phase1/                  # DeepLabV3+ Weights<br>
│   └── ... (other phases)<br>
├── web_interface/<br>
│   ├── app.py                   # Flask Server Entrypoint<br>
│   ├── static/                  # CSS, JS, Uploads<br>
│   └── templates/               # HTML (Inference & Analysis Pages)<br>
├── Dockerfile                   # Production Docker Configuration<br>
└── requirements.txt             # Python Dependencies<br>

## 🤝 Acknowledgements<br>

***Requestly:** For providing the API client used to debug and test our inference endpoints.

***Hugging Face:** For hosting the model weights and the live demo space.

***Timm Library:** For the efficient model backbones.
