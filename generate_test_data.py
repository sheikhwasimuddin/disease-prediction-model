import pandas as pd
import numpy as np

# Set random seed
np.random.seed(42)

# Define test data
test_data = {
    'Patient Number': [1, 2, 3, 4, 5],
    'Heart Rate (bpm)': [75, 110, 95, 50, 85],
    'SpO2 Level (%)': [98, 90, 92, 96, 97],
    'Systolic Blood Pressure (mmHg)': [115, 125, 160, 130, 140],
    'Diastolic Blood Pressure (mmHg)': [75, 80, 100, 85, 90],
    'Body Temperature (°C)': [36.8, 37.0, 37.1, 36.9, 37.8],
    'Fall Detection': ['No', 'Yes', 'No', 'No', 'Yes'],
    'Heart Rate Alert': ['Normal', 'High', 'Normal', 'Low', 'Normal'],
    'SpO2 Level Alert': ['Normal', 'Low', 'Low', 'Normal', 'Normal'],
    'Blood Pressure Alert': ['Normal', 'Normal', 'High', 'Normal', 'High'],
    'Temperature Alert': ['Normal', 'Normal', 'Normal', 'Normal', 'Abnormal'],
    'Data Accuracy (%)': [95, 90, 92, 88, 93]
}

# Create DataFrame
df_test = pd.DataFrame(test_data)

# Save to Excel
df_test.to_excel('new_patients_data.xlsx', index=False)
print("Test dataset saved to 'new_patients_data.xlsx'")