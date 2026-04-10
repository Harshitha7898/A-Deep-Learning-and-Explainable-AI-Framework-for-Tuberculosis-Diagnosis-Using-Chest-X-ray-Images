# A Deep Learning and Explainable AI Framework for Tuberculosis Diagnosis Using Chest X-ray Images

This repository contains a web application for tuberculosis detection using deep learning models and explainability techniques.

## Contents

- `app.py` - Main Flask application for the web interface
- `predict.py` - Prediction helper functions
- `model_loader.py` - Model loading utilities
- `gradcam.py`, `occlusion.py`, `captum_explain.py` - Explainability utilities
- `frontend/` - HTML pages for the user interface
- `requirements.txt` - Python dependencies

## Setup

1. Create a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open the web interface in your browser.

## Notes

- The `models/` folder contains trained `.pth` model files that are large and may not be pushed to GitHub.
- If you need to deploy, ensure the model files are available locally or hosted separately.
