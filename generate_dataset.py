import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define number of samples per class
n_samples_per_class = 10000
n_classes = 5
total_samples = n_samples_per_class * n_classes

# Define disease-specific feature ranges
disease_ranges = {
    'Normal': {
        'Heart Rate (bpm)': (60, 100),
        'SpO2 Level (%)': (95, 100),
        'Systolic Blood Pressure (mmHg)': (100, 120),
        'Diastolic Blood Pressure (mmHg)': (60, 80),
        'Body Temperature (°C)': (36.5, 37.2),
    },
    'Asthma': {
        'Heart Rate (bpm)': (80, 120),
        'SpO2 Level (%)': (85, 94),
        'Systolic Blood Pressure (mmHg)': (100, 130),
        'Diastolic Blood Pressure (mmHg)': (60, 85),
        'Body Temperature (°C)': (36.5, 37.5),
    },
    'Hypertension': {
        'Heart Rate (bpm)': (70, 110),
        'SpO2 Level (%)': (90, 98),
        'Systolic Blood Pressure (mmHg)': (140, 180),
        'Diastolic Blood Pressure (mmHg)': (90, 110),
        'Body Temperature (°C)': (36.5, 37.2),
    },
    'Arrhythmia': {
        'Heart Rate (bpm)': (40, 140),
        'SpO2 Level (%)': (90, 98),
        'Systolic Blood Pressure (mmHg)': (100, 140),
        'Diastolic Blood Pressure (mmHg)': (60, 90),
        'Body Temperature (°C)': (36.5, 37.2),
    },
    'Diabetes Mellitus': {
        'Heart Rate (bpm)': (70, 110),
        'SpO2 Level (%)': (90, 98),
        'Systolic Blood Pressure (mmHg)': (120, 150),
        'Diastolic Blood Pressure (mmHg)': (80, 100),
        'Body Temperature (°C)': (37.0, 38.0),
    }
}

# Initialize data
data = {
    'Patient Number': np.arange(1, total_samples + 1),
    'Heart Rate (bpm)': np.zeros(total_samples),
    'SpO2 Level (%)': np.zeros(total_samples),
    'Systolic Blood Pressure (mmHg)': np.zeros(total_samples),
    'Diastolic Blood Pressure (mmHg)': np.zeros(total_samples),
    'Body Temperature (°C)': np.zeros(total_samples),
    'Fall Detection': np.random.choice(['No', 'Yes'], total_samples, p=[0.95, 0.05]),
    'Heart Rate Alert': np.zeros(total_samples, dtype=object),
    'SpO2 Level Alert': np.zeros(total_samples, dtype=object),
    'Blood Pressure Alert': np.zeros(total_samples, dtype=object),
    'Temperature Alert': np.zeros(total_samples, dtype=object),
    'Data Accuracy (%)': np.random.uniform(85, 100, total_samples),
    'Predicted Disease': np.repeat(list(disease_ranges.keys()), n_samples_per_class)
}

# Generate numerical features and alerts
for i, disease in enumerate(data['Predicted Disease']):
    ranges = disease_ranges[disease]
    data['Heart Rate (bpm)'][i] = np.random.uniform(*ranges['Heart Rate (bpm)'])
    data['SpO2 Level (%)'][i] = np.random.uniform(*ranges['SpO2 Level (%)'])
    data['Systolic Blood Pressure (mmHg)'][i] = np.random.uniform(*ranges['Systolic Blood Pressure (mmHg)'])
    data['Diastolic Blood Pressure (mmHg)'][i] = np.random.uniform(*ranges['Diastolic Blood Pressure (mmHg)'])
    data['Body Temperature (°C)'][i] = np.random.uniform(*ranges['Body Temperature (°C)'])

    # Generate alerts based on thresholds
    data['Heart Rate Alert'][i] = 'Normal' if 60 <= data['Heart Rate (bpm)'][i] <= 100 else 'High' if data['Heart Rate (bpm)'][i] > 100 else 'Low'
    data['SpO2 Level Alert'][i] = 'Normal' if data['SpO2 Level (%)'][i] >= 95 else 'Low'
    data['Blood Pressure Alert'][i] = 'Normal' if 100 <= data['Systolic Blood Pressure (mmHg)'][i] <= 120 and 60 <= data['Diastolic Blood Pressure (mmHg)'][i] <= 80 else 'High'
    data['Temperature Alert'][i] = 'Normal' if 36.5 <= data['Body Temperature (°C)'][i] <= 37.2 else 'Abnormal'

# Create DataFrame
df = pd.DataFrame(data)

# Introduce missing values (~5%)
for col in ['Heart Rate (bpm)', 'SpO2 Level (%)', 'Systolic Blood Pressure (mmHg)', 
            'Diastolic Blood Pressure (mmHg)', 'Body Temperature (°C)']:
    mask = np.random.choice([True, False], total_samples, p=[0.05, 0.95])
    df.loc[mask, col] = np.nan

# Save to Excel
df.to_excel('patients_data_with_alerts.xlsx', index=False)
print("Synthetic dataset saved to 'patients_data_with_alerts.xlsx'")