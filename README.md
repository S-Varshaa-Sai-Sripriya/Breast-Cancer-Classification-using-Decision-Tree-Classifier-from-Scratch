# 🧬 Breast Cancer Classification using Decision Tree Classifier from Scratch

This project implements a Decision Tree Classifier from scratch to classify tumors as Malignant or Benign using the Breast Cancer Wisconsin dataset.  

It demonstrates a complete Machine Learning workflow — from data ingestion and preprocessing to model training, evaluation, and metrics visualization — all modularized for clarity and reusability.

---

## ⚙️ Features
- 📥 **Data Ingestion** – Load breast cancer dataset directly.  
- 🧹 **Preprocessing** – Handle categorical features, normalization, and train-test split.  
- 🌳 **Decision Tree (Scratch Implementation)** – No sklearn `DecisionTreeClassifier`.  
- 📊 **Evaluation** – Accuracy, predictions vs. actuals, metrics JSON export.  
- 📝 **Logging** – Step-by-step execution logs for traceability.  

---

## 📊 Dataset Explanation:

### Breast Cancer Dataset
The Breast Cancer Wisconsin dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses.

- **Number of Instances**: 569  
- **Number of Features**: 30 + 1 target  
- **Target Variable**: `diagnosis` (malignant or benign)

**Feature Details**:

- `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, ... – Statistical measurements of cell nuclei.  
- `diagnosis` – Target class indicating whether the tumor is malignant (M) or benign (B).

This dataset is useful for:

- Classification tasks for medical diagnosis.  
- Benchmarking machine learning models in healthcare applications.  
---

Results

Accuracy: ~93.8%
