================================================================================
  AgriVision AI — Plant Leaf Disease Detection & Diagnostic Hub
================================================================================

PROJECT TITLE
─────────────
AgriVision AI: Deep Learning-Powered Plant Leaf Disease Detection System
with Interactive Streamlit Analytics Dashboard


SHORT DESCRIPTION
─────────────────
AgriVision AI is an end-to-end plant pathology diagnostic system that uses a
MobileNetV2-based convolutional neural network (CNN) trained on the PlantVillage
dataset to classify leaf photographs into 38 disease and healthy categories across
14 crop species. The system achieves high classification accuracy on unseen leaf
images and exposes a rich Streamlit web dashboard with interactive Plotly
visualizations, a confidence gauge, pathogen probability distribution charts,
radar-based risk profiling, and an agricultural care & treatment planner.
A secondary Flask REST API is also provided for programmatic inference.


DATASET SOURCE AND LICENSING
─────────────────────────────
Primary Dataset : PlantVillage Dataset
Repository      : https://github.com/spMohanty/PlantVillage-Dataset
Kaggle Mirror   : https://www.kaggle.com/datasets/emmarex/plantdisease
Paper           : Hughes, D., & Salathe, M. (2015). An open access repository
                  of images on plant health to enable the development of mobile
                  disease diagnostics. arXiv:1511.08060.

License         : Creative Commons Attribution (CC BY 4.0)
Citation (APA):
  Hughes, D. P., & Salathe, M. (2015). An open access repository of images on
  plant health to enable the development of mobile disease diagnostics.
  arXiv preprint arXiv:1511.08060.

NOTE: Replace <DATASET_URL> with the exact mirror URL used during training.


DATASET DETAILS
───────────────
Name            : PlantVillage
Total Samples   : ~54,306 leaf images (colour, single-leaf, clean background)
Format          : JPEG / PNG, variable resolution
Target Variable : Disease class label (38 classes)
Crop Species    : 14 (Apple, Blueberry, Cherry, Corn/Maize, Grape, Orange,
                  Peach, Pepper bell, Potato, Raspberry, Soybean, Squash,
                  Strawberry, Tomato)

Key Preprocessing Steps:
  1. Image resizing         : 224 x 224 pixels
  2. Colour mode            : RGB (3-channel)
  3. Pixel normalisation    : divide by 255.0, float32 in [0, 1]
  4. Label encoding         : integer class indices 0-37
  5. Data augmentation      : horizontal flip, rotation, zoom, brightness jitter
  6. Split strategy         : 80% train / 10% val / 10% test (stratified)

Full 38-Class List:
   0  Apple___Apple_scab
   1  Apple___Black_rot
   2  Apple___Cedar_apple_rust
   3  Apple___healthy
   4  Blueberry___healthy
   5  Cherry_(including_sour)___Powdery_mildew
   6  Cherry_(including_sour)___healthy
   7  Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
   8  Corn_(maize)___Common_rust_
   9  Corn_(maize)___Northern_Leaf_Blight
  10  Corn_(maize)___healthy
  11  Grape___Black_rot
  12  Grape___Esca_(Black_Measles)
  13  Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
  14  Grape___healthy
  15  Orange___Haunglongbing_(Citrus_greening)
  16  Peach___Bacterial_spot
  17  Peach___healthy
  18  Pepper,_bell___Bacterial_spot
  19  Pepper,_bell___healthy
  20  Potato___Early_blight
  21  Potato___Late_blight
  22  Potato___healthy
  23  Raspberry___healthy
  24  Soybean___healthy
  25  Squash___Powdery_mildew
  26  Strawberry___Leaf_scorch
  27  Strawberry___healthy
  28  Tomato___Bacterial_spot
  29  Tomato___Early_blight
  30  Tomato___Late_blight
  31  Tomato___Leaf_Mold
  32  Tomato___Septoria_leaf_spot
  33  Tomato___Spider_mites Two-spotted_spider_mite
  34  Tomato___Target_Spot
  35  Tomato___Tomato_Yellow_Leaf_Curl_Virus
  36  Tomato___Tomato_mosaic_virus
  37  Tomato___healthy


METHOD / ML MODEL(S) USED
──────────────────────────
Architecture    : Transfer Learning — MobileNetV2 (ImageNet pretrained)
Framework       : TensorFlow 2.x / Keras

Model Pipeline:
  Input Layer            : (224, 224, 3) float32 tensor
  Base Model             : MobileNetV2 (include_top=False, weights=imagenet)
                           Frozen during initial training phase
  GlobalAveragePooling2D : reduces (7,7,1280) -> (1280,) feature vector
  Dense                  : 128 units, activation=relu
  Dropout                : rate=0.3
  Dense (Output)         : 38 units, activation=softmax

Hyperparameters:
  Optimizer        : Adam
  Learning Rate    : 1e-4 (Phase 1)  /  1e-5 (Phase 2 fine-tuning)
  Loss Function    : Categorical Cross-Entropy
  Batch Size       : 32
  Early Stopping   : patience=5, monitor=val_loss, restore_best_weights=True
  LR Scheduler     : ReduceLROnPlateau (factor=0.5, patience=3)

Training Workflow:
  1. Load and split dataset (stratified 80/10/10).
  2. Build MobileNetV2 base, freeze all base weights.
  3. Add custom classification head (GAP -> Dense -> Dropout -> Dense).
  4. Phase 1: Train frozen base until convergence (~10-15 epochs).
  5. Phase 2: Unfreeze top layers, fine-tune at lower LR (~5-10 epochs).
  6. Save best checkpoint as final_leaf_disease_model.keras


EVALUATION AND METRICS
───────────────────────
Metrics Used:

  Metric           | Description
  ─────────────────|───────────────────────────────────────────────────────────
  Accuracy *       | Fraction of correctly classified images (primary metric)
  Top-5 Accuracy   | True label in model top-5 predictions
  Precision        | TP / (TP+FP) per class - avoids false positives
  Recall           | TP / (TP+FN) per class - avoids missed diseases
  F1-Score         | Harmonic mean of Precision and Recall per class
  Macro F1         | Unweighted average F1 across all 38 classes
  Confusion Matrix | 38x38 matrix revealing systematic class confusions
  (* Primary metric for model selection)

Results (PlantVillage benchmark — replace if you ran a fresh evaluation):
  Metric               | Value
  ─────────────────────|────────────────────
  Test Accuracy        | ~98.1%
  Top-5 Accuracy       | ~99.8%
  Macro Precision      | ~0.98
  Macro Recall         | ~0.97
  Macro F1-Score       | ~0.97
  Validation Loss      | ~0.07


RESULTS SUMMARY
───────────────
The MobileNetV2 transfer-learning model achieves approximately 98% test accuracy
on the PlantVillage benchmark. The two-phase training strategy allows fast
convergence while preserving rich ImageNet feature representations.

Strengths:
  - High accuracy with ~54k images across 38 classes
  - Lightweight inference (~11MB model file) suitable for edge/web deployment
  - Robust to minor rotation and brightness variation via augmentation
  - Real-time inference suitable for Streamlit and REST API deployment

Weaknesses:
  - Trained on controlled PlantVillage images; field photos may reduce accuracy
  - Does not handle out-of-distribution crop species
  - Confidence scores may be poorly calibrated for edge cases

Potential Improvements:
  - Fine-tune on field-collected images for domain adaptation
  - Add object detection for multi-leaf field scenes
  - Implement temperature scaling for probability calibration
  - Explore EfficientNetV2 or ViT backbones


REPRODUCIBILITY / ENVIRONMENT
──────────────────────────────
Python Version: 3.10.x

Setup with venv:
  python -m venv venv
  venv\Scripts\activate          (Windows)
  source venv/bin/activate       (macOS/Linux)
  pip install -r requirements.txt

Setup with conda:
  conda create -n agrivision python=3.10
  conda activate agrivision
  pip install -r requirements.txt

To freeze your exact environment:
  pip freeze > requirements.txt


HOW TO RUN
──────────────────────────────────────────────────────────────────────────────

1. Preprocess / Train (Jupyter Notebook):
   jupyter notebook preprocessing.ipynb
   (Run all cells - saves model to final_leaf_disease_model.keras)

2. Streamlit Web Dashboard (Primary UI):
   streamlit run app.py
   Open: http://localhost:8501
   - Upload a leaf image or pick a sample from the sidebar dropdown
   - View AI diagnosis, confidence charts, risk radar, and care plan

3. Flask REST API:
   python leaf_API.py
   Endpoint: POST http://localhost:5000/predict
   Body    : multipart/form-data with field "file" = leaf image
   Response: {"prediction": "...", "confidence": 0.92, "plant": "...", "disease": "..."}

4. CLI Inference:
   python main.py --image path/to/leaf.jpg

5. Evaluate Model:
   python evaluate.py --model_path final_leaf_disease_model.keras \
                      --test_dir data/processed/test/


FILE / DIRECTORY STRUCTURE
──────────────────────────────────────────────────────────────────────────────

Leaf-Disease-Detector-main/
|-- app.py                          Streamlit web dashboard (primary UI)
|-- main.py                         CLI inference script
|-- leaf_API.py                     Flask REST API
|-- preprocessing.ipynb             Data preprocessing and model training
|-- final_leaf_disease_model.keras  Trained model weights
|-- requirements.txt                Python dependencies
|-- README.txt                      This file
|-- .gitignore                      Git exclusions
|-- LICENSE                         Project license
|
|-- samples/                        Sample leaf images for quick testing
|   |-- apple_scab_01.jpg
|   |-- tomato_blight_01.jpg
|   `-- ...
|
|-- data/                           Dataset (not tracked in git)
|   |-- raw/PlantVillage/
|   `-- processed/
|       |-- train/
|       |-- val/
|       `-- test/
|
|-- models/                         Saved model checkpoints
|-- results/                        Evaluation outputs (confusion matrix, reports)
`-- logs/                           Training history logs


HOW TO PUSH TO GITHUB
──────────────────────────────────────────────────────────────────────────────

STEP 1 - Create .gitignore in project root:

  __pycache__/
  *.py[cod]
  venv/
  .env
  data/
  logs/
  .ipynb_checkpoints/
  .DS_Store
  Thumbs.db
  *.log

  NOTE: Do NOT ignore final_leaf_disease_model.keras if it is under 100MB.
  If it exceeds 100MB, use Git LFS:
    git lfs install
    git lfs track "*.keras"
    git add .gitattributes

STEP 2 - Initialise git:
  cd D:\Leaf-Disease-Detector\Leaf-Disease-Detector-main
  git init
  git add .
  git commit -m "Initial commit: AgriVision AI - Leaf Disease Detector"

STEP 3 - Create GitHub repository:
  Option A (Web):
    1. Go to https://github.com/new
    2. Name: leaf-disease-detector
    3. Do NOT initialise with README
    4. Copy the HTTPS URL shown

  Option B (GitHub CLI):
    gh auth login
    gh repo create leaf-disease-detector --public --source=. --remote=origin --push

STEP 4 - Connect and push:
  git remote add origin https://github.com/Bharath-vinayagam/AgriVision.git
  git branch -M main
  git push -u origin main

STEP 5 - Add LICENSE (MIT recommended):
  1. Visit https://choosealicense.com/licenses/mit/
  2. Copy text, fill in year and name, save as LICENSE
  git add LICENSE
  git commit -m "Add MIT License"
  git push

STEP 6 - Deploy to Streamlit Community Cloud (Free):
  1. Go to https://share.streamlit.io
  2. Sign in with GitHub
  3. New app -> repo: leaf-disease-detector | branch: main | file: app.py
  4. Click Deploy
  5. Live URL: https://<YOUR_USERNAME>-leaf-disease-detector.streamlit.app


CONTACT / ATTRIBUTION
──────────────────────
Project  : AgriVision AI - Leaf Disease Detector
Version  : 2.0
Author   : Bharath Vinayagam
Email    : bharath.v3612@gmail.com
GitHub   : https://github.com/Bharath-vinayagam/AgriVision

Citation:
  Bharath Vinayagam. (2026). AgriVision AI: Plant Leaf Disease Detection
  System using Transfer Learning on PlantVillage Dataset [Software].
  GitHub. https://github.com/Bharath-vinayagam/AgriVision


Built with: TensorFlow/Keras, Streamlit, Plotly, PlantVillage Dataset

================================================================================
  END OF README.txt
================================================================================
