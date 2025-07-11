# 🩺 Disease Prediction Using Patient Health Data

This project uses synthetic patient health data to train a machine learning model that predicts potential diseases based on vital signs and alert indicators. It simulates an IoT-based health monitoring environment where patient data is collected remotely and analyzed in real time.

---

## 📘 Project Description

The system takes vital parameters such as heart rate, SpO₂ level, blood pressure, and body temperature, along with alert signals like fall detection or abnormal values. Based on these, it predicts the most likely disease using a trained neural network model.

### 🔍 Diseases Covered:
- Normal (Healthy)
- Asthma
- Hypertension
- Arrhythmia
- Diabetes Mellitus

---
```
## 📁 Project Structure

disease-prediction-iot/
│
├── data/
│ ├── patients_data_with_alerts.xlsx # Synthetic training dataset
│ └── new_patients_data.xlsx # Example test dataset
│
├── models/
│ ├── disease_prediction_model.keras # Trained Keras model
│ ├── scaler.pkl # Scaler for standardizing inputs
│ ├── label_encoder.pkl # Label encoder for diseases
│ └── feature_names.pkl # Feature list used during training
│
├── generate_dataset.py # Generate synthetic dataset
├── generate_test_data.py # Generate small test dataset
├── train_disease_model.py # Train model using synthetic data
├── predict_disease.py # Predict disease for new patients
│
├── requirements.txt # Required Python packages
└── README.md # This documentation file
```

---

## 🧠 How It Works

1. **`generate_dataset.py`**  
   Creates a large synthetic dataset (50,000 samples) with realistic patient data based on disease-specific ranges.

2. **`train_disease_model.py`**  
   Preprocesses data, performs feature engineering, trains a neural network, and saves the model and encoders.

3. **`predict_disease.py`**  
   Loads new patient data, applies the same preprocessing, and outputs disease predictions.

4. **`generate_test_data.py`**  
   Creates a smaller dataset with sample values for quick testing.

---

## 📦 Installation

```bash
pip install -r requirements.txt
Python version: Recommended 3.8 or later.

🏁 Running the Project
1. Generate Synthetic Dataset
python generate_dataset.py
2. Train the Model
python train_disease_model.py
3. Predict Disease on New Data
python predict_disease.py
```
🧪 Sample Output (CSV)
The output file patient_disease_predictions.csv contains:

### Patient ID,Predicted Disease
### 1,Hypertension
### 2,Asthma
### 3,Normal
### 4,Arrhythmia
### 5,Diabetes Mellitus
### ., .....
---
📊 Model Details
Architecture: 4-layer neural network

Framework: TensorFlow / Keras

Input Features: Vitals, alerts, engineered ratios

Output: Multi-class softmax prediction

---
⚠️ Notes
The dataset is fully synthetic and randomly generated.

Make sure the Excel files (.xlsx) exist before training/predicting.

All alerts are rule-based based on clinical thresholds.
---
📄 License
This project is for educational and experimental use only.
MANIT,BHOPAL.

🙋‍♂️ Author
Sheikh Wasimuddin,Himanshu
Yeshwantrao Chavan College of Engineering, Nagpur|| Central University of jharkhand

