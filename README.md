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

## 📊 Dataset
- **Breast Cancer Wisconsin Dataset** from UCI ML Repository / sklearn.  
- Classes:  
  - `0` → Malignant (Cancerous)  
  - `1` → Benign (Non-cancerous)  
- **Features**: 30 numeric features (mean, SE, worst of cell nuclei measurements).  

---

Results

Accuracy: ~93.8%
