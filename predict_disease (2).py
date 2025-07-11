import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the saved model, scaler, and label encoder
model = load_model('disease_prediction_model.keras')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Load expected feature names from training (saved during training)
try:
    expected_features = joblib.load('feature_names.pkl')
except FileNotFoundError:
    print("Error: feature_names.pkl not found. Please ensure it was saved during training.")
    raise

# Print LabelEncoder classes for debugging
print("LabelEncoder classes:", le.classes_.tolist())
print("Expected features from training:", expected_features)

# Load new patient data
new_data = pd.read_excel('/content/patients_data_with_alerts.xlsx')

# Print column names for debugging
print("Columns in new_patients_data.xlsx:", new_data.columns.tolist())

# Store patient identifiers
patient_ids = new_data.get('Patient Number', pd.Series(range(1, len(new_data) + 1)))

# Feature engineering
new_data['BP_Difference'] = new_data['Systolic Blood Pressure (mmHg)'] - new_data['Diastolic Blood Pressure (mmHg)']
new_data['Heart_Rate_SpO2_Ratio'] = new_data['Heart Rate (bpm)'] / new_data['SpO2 Level (%)']

# Drop irrelevant columns
new_data = new_data.drop(['Patient Number', 'Data Accuracy (%)'], axis=1, errors='ignore')

# One-hot encode categorical features
categorical_cols = ['Fall Detection', 'Heart Rate Alert', 'SpO2 Level Alert', 
                    'Blood Pressure Alert', 'Temperature Alert']
new_data = pd.get_dummies(new_data, columns=categorical_cols, drop_first=True)

# Print columns after preprocessing
print("Columns after preprocessing:", new_data.columns.tolist())

# Handle missing values
numerical_cols = ['Heart Rate (bpm)', 'SpO2 Level (%)', 'Systolic Blood Pressure (mmHg)', 
                  'Diastolic Blood Pressure (mmHg)', 'Body Temperature (°C)', 
                  'BP_Difference', 'Heart_Rate_SpO2_Ratio']
new_data[numerical_cols] = new_data[numerical_cols].fillna(new_data[numerical_cols].median())

# Scale numerical features
new_data[numerical_cols] = scaler.transform(new_data[numerical_cols])

# Align columns with training
missing_cols = [col for col in expected_features if col not in new_data.columns]
for col in missing_cols:
    new_data[col] = 0
extra_cols = [col for col in new_data.columns if col not in expected_features]
if extra_cols:
    print(f"Warning: Dropping extra columns not seen in training: {extra_cols}")
    new_data = new_data.drop(extra_cols, axis=1)
new_data = new_data[expected_features]  # Reorder columns

# Print final shape for debugging
print("Final input shape:", new_data.shape)

# Make predictions
predictions = model.predict(new_data)
predicted_classes = np.argmax(predictions, axis=1)
predicted_diseases = le.inverse_transform(predicted_classes)

# Create output DataFrame
output = pd.DataFrame({
    'Patient ID': patient_ids,
    'Predicted Disease': predicted_diseases
})

# Display predictions
print("\nPredictions for New Patients:")
print(output)

# Save predictions
output.to_csv('patient_disease_predictions.csv', index=False)
print("\nPredictions saved to 'patient_disease_predictions.csv'")