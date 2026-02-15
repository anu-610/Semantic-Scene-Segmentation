# Semantic Scene Segmentation

A multi-phase semantic segmentation repository where phase1–phase6 are independent training/evaluation workspaces (e.g., SegFormer, UNet++, FPN) and the Ensemble directory is the “command center” that combines their best `.pth` weights into a more robust, generalized prediction for your hackathon or competition.

## Project overview

The project is structured around a multi-phase training and evaluation strategy:
- Each `phaseX` directory holds a separate architecture and experiment (SegFormer, EfficientNet, UNet++, FPN, etc.).
- Models in each phase train independently and save their best checkpoint as a `.pth` file (e.g., `best_model_phase4_segformer.pth`).
- The `Ensemble/` directory then:
  - Loads multiple `.pth` weights from different phases.
  - Performs a unified inference (weighted averaging or voting-based ensemble).
  - Outputs the final submission results and a full evaluation report.

In short:  
1) Train strong models in phases → 2) Save `.pth` files → 3) Run `Ensemble/inference.py` → 4) Get boosted submission + graphs.

## Core components

### Ensemble Engine (main component)

- Located in `Ensemble/`.
- Python script: `inference.py`.
- It is the main entry point for the final project output.
- Loads `.pth` weights from various phases (SegFormer, EfficientNet, UNet++, FPN, etc.).
- Blends predictions using either:
  - weighted averaging of probability maps, or
  - voting-based combination (hard or soft voting).
- Writes final image/CSV predictions and a full report to:
  - `Ensemble/Submission_Results_Boosted_Full_Report/`.

### Analytical outputs

For each phase and for the final ensemble, the system generates:

- **Submission results**
  - CSV files and/or image mask predictions suitable for evaluation or submission to a leaderboard.
- **Performance metrics**
  - Confusion matrices (class-wise confusion).
  - mIoU (Mean Intersection over Union) graphs over epochs.
  - Class-wise IoU and accuracy.
  - Training/validation loss graphs to track model convergence.

These artifacts help you identify which classes the model is struggling with (e.g., distinguishing “Road” vs “Sidewalk”) and guide the next iteration (loss design, class weighting, augmentation, etc.).

### Modular training (phase1 to phase6)

- Each phase directory is self-contained:
  - `phase1/`
  - `phase2/`
  - ...
  - `phase6/`
- Inside each phase you typically find:
  - `train.py`: trains the model for that phase and saves the best checkpoint.
  - `test.py`: validates the model on held-out data and generates metrics.
  - `model_utils.py`: utility functions for:
    - data augmentation pipelines (using Albumentations),
    - loss functions (e.g., Dice Loss, Focal Loss, or custom combinations),
    - metric calculations (mIoU, confusion matrices, class-wise stats).

Because each phase is isolated, you can:
- Tune hyperparameters independently.
- Try different architectures or backbones.
- Run ablation studies without breaking anything in other phases.

## Directory breakdown

Here is the expected layout when you open the project.

Description table:

| Directory / File                    | Short description |
|-------------------------------------|-------------------|
| `Ensemble/`                        | Main ensemble component; runs multi-model inference. |
| `Ensemble/inference.py`            | Main script that loads `.pth` files and blends predictions. |
| `Ensemble/Submission_Results_Boosted_Full_Report/` | Final competition/project output folder (predictions + full report). |
| `phase1/` ... `phase6/`            | Independent training environments for different architectures. |
| `phaseX/train.py`                  | Training script for phase `X`. |
| `phaseX/test.py`                   | Validation script for phase `X` (evaluation and metrics). |
| `phaseX/model_utils.py`            | Shared utilities: augmentation, loss functions, metrics. |
| `*.pth` files (e.g., `best_model_phase4_segformer.pth`) | Pre-trained weights for specific architectures, loaded by the Ensemble engine. |

## Setup & installation

### Environment setup

Requirements:
- Python 3.8+
- CUDA + NVIDIA GPU (recommended for training speed)

Recommended: use a virtual environment.

```bash
# Create virtual environment
python -m venv env

# Activate (Linux/macOS)
source env/bin/activate

# Activate (Windows)
env\Scripts\activate

Install dependencies

# Core PyTorch stack
pip install torch torchvision torchaudio

# Segmentation and image utilities
pip install segmentation-models-pytorch albumentations opencv-python matplotlib pandas

Running instructions-

1. Training individual phases
cd phase5
python train.py

2. Running the Ensemble (the main thing)
cd Ensemble
python inference.py

3. Evaluation & metrics (where to look)
After running train.py in any phase or inference.py in Ensemble/, the performance visualizations are stored in:

Phase‑specific folders:

phaseX/Submission_Results/ (or phaseX/results/, depending on your code)

Final ensemble report:

Ensemble/Submission_Results_Boosted_Full_Report/

Inside these folders you will find:

Confusion matrices
Show which classes are getting confused (e.g., “Road” vs “Sidewalk”).

mIoU graphs
Track mean Intersection over Union over epochs to see convergence.

Class-wise IoU and accuracy
Help you identify which classes are weak and which are strong.

Loss graphs
Show training and validation loss trends.

Copyright (c) 2026
All rights reserved.