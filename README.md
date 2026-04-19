# AI Framework for Tuberculosis Diagnosis Using Chest X-ray Images

##  Overview

Tuberculosis (TB) is a life-threatening infectious disease that primarily affects the lungs. Early detection is essential to reduce transmission and mortality. This project presents an **AI-based framework** that automatically detects tuberculosis from chest X-ray images using advanced deep learning models and Explainable AI (XAI) techniques.

The system leverages state-of-the-art architectures such as **ConvNeXt, Vision Transformer (ViT), Swin Transformer**, and an **ensemble model**, combined with interpretability methods to improve trust in medical diagnosis.

---

## Objectives

* Develop an automated system for TB detection from chest X-rays
* Compare performance of multiple deep learning models
* Improve accuracy using ensemble learning
* Integrate Explainable AI techniques for model interpretability
* Assist healthcare professionals in decision-making

---

##  Models Used

* ConvNeXt-Tiny
* Vision Transformer (ViT)
* Swin Transformer
* Ensemble Model (ConvNeXt + ViT)

---

## 📂 Dataset

* **Source:** Kaggle TB Chest X-ray Dataset

* **Classes:**

  * Normal
  * Tuberculosis

* **Dataset Structure:**

```
TB_Data/
 ┣ Train/
 ┃ ┣ Normal/
 ┃ ┗ Tuberculosis/
 ┣ Validation/
 ┃ ┣ Normal/
 ┃ ┗ Tuberculosis/
 ┣ Test/
   ┣ Normal/
   ┗ Tuberculosis/
```


---

## Technologies Used

* Python
* PyTorch
* Torchvision
* OpenCV
* Matplotlib
* Scikit-learn
* Kaggle (for GPU training)

---

## Workflow

1. Data preprocessing and augmentation
2. Model loading with pretrained weights
3. Training with weighted loss to handle imbalance
4. Validation and early stopping
5. Performance evaluation using metrics
6. Explainability using XAI techniques

---

## Performance Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC Curve
* Confusion Matrix

---

## Results Summary

| Model            | Accuracy | Precision | Recall | F1-Score |
| ---------------- | -------- | --------- | ------ | -------- |
| ConvNeXt         | 97%      | 99%       | 99%    | 97%      |
| ViT              | 99%      | 99%       | 99%    | 99%      |
| Swin Transformer | 99%      | 99%       | 99%    | 99%      |
| Ensemble         | 98%      | 98%       | 98%    | 98%      |


---

## Explainable AI (XAI)

To improve model transparency, the following techniques are used:

* Grad-CAM / Grad-CAM++
* Captum
* Occlusion

These methods highlight important regions in X-ray images, helping to understand how the model makes predictions and ensuring reliability in medical applications.

---


---

## Future Work

* Improve model performance using hybrid architectures
* Apply advanced ensemble strategies
* Deploy as a web application using Streamlit/Flask
* Integrate real-time clinical data
* Extend to multi-disease detection

---

## Conclusion

This project demonstrates the effectiveness of deep learning models in detecting tuberculosis from chest X-rays. The integration of Explainable AI enhances trust and interpretability, making the system suitable for real-world healthcare applications.

---

---


## Usage

Run the notebooks or scripts to:

* Train models
* Evaluate performance
* Generate XAI visualizations

---

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
