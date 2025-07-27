# 🧪 Higgs Boson Event Classification with Deep Neural Network (ANN)

This project builds a deep learning classifier to distinguish between real Higgs boson particle collision events (signal) and background noise, using the dataset from the [Kaggle Higgs Boson Machine Learning Challenge](https://www.kaggle.com/c/higgs-boson).

---

## 🚀 Overview

The dataset contains engineered physics features describing high-energy particle collisions. The objective is to classify each event as either a **signal (Higgs boson)** or **background**.

This project demonstrates:

- Deep learning applied to tabular scientific data
- Handling of missing/imputed values (`-999`)
- Advanced regularization using Dropout + BatchNorm
- Threshold tuning for best F1-score
- High AUC performance despite class imbalance

---

## 📁 Dataset

- 📄 File: `training.csv`
- 💡 Features: 28 numerical physics attributes
- 🎯 Target: `Label` column — `s` (signal) or `b` (background)
- 🧪 Source: CERN via Kaggle

---

## 🧠 Model Architecture

A fully-connected ANN (Feedforward Neural Network):

- Input: 28 features
- Dense(700) → BatchNorm → Dropout(0.5)
- Dense(500) → BatchNorm → Dropout(0.3)
- Dense(250) → BatchNorm
- Dense(100) → BatchNorm
- Dense(50) → Dropout(0.1)
- Dense(25) → BatchNorm
- Output: Dense(1, activation='sigmoid')

**Loss**: Binary Crossentropy  
**Optimizer**: Adam  
**Metrics**: Accuracy, AUC  
**Class Weights**: `{0: 1.0, 1: 2.0}`  
**EarlyStopping**: Based on `val_auc`

---

## 📊 Results

| Metric       | Validation Set | Test Set |
|--------------|----------------|----------|
| **Accuracy** | 0.834          | 0.830    |
| **AUC**      | 0.913          | 0.910    |
| **F1-Score** (signal class) | 0.75     | 0.75     |

✅ Model shows strong generalization  
✅ Balanced performance on both classes  
✅ Threshold optimized to `0.67` for improved precision/recall trade-off

---

## 🧪 How to Run

```bash
# 1. Clone this repo
git clone https://github.com/<your-username>/higgs-boson-event-classification.git
cd higgs-boson-event-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook notebooks/Higgs_Boson_Classifier.ipynb
