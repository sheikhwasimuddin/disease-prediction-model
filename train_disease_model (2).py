import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_excel('/content/patients_data_with_alerts.xlsx')

# Print data diagnostics
print("Class distribution:\n", df['Predicted Disease'].value_counts(normalize=True))
print("Missing values:\n", df.isnull().sum())
print("Data summary:\n", df.describe())
print("Unique values in categorical columns:")
categorical_cols = ['Fall Detection', 'Heart Rate Alert', 'SpO2 Level Alert', 
                    'Blood Pressure Alert', 'Temperature Alert', 'Predicted Disease']
for col in categorical_cols:
    print(f"{col}: {df[col].unique()}")

# Check feature-target correlations
numerical_cols = ['Heart Rate (bpm)', 'SpO2 Level (%)', 'Systolic Blood Pressure (mmHg)', 
                  'Diastolic Blood Pressure (mmHg)', 'Body Temperature (°C)']
for col in numerical_cols:
    correlation = df[col].corr(df['Predicted Disease'].astype('category').cat.codes)
    print(f"Correlation of {col} with Predicted Disease: {correlation:.4f}")

# Handle missing values
if df.isnull().any().any():
    print("Warning: Missing values detected. Filling with defaults.")
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Feature engineering
df['BP_Difference'] = df['Systolic Blood Pressure (mmHg)'] - df['Diastolic Blood Pressure (mmHg)']
df['Heart_Rate_SpO2_Ratio'] = df['Heart Rate (bpm)'] / df['SpO2 Level (%)']

# Drop irrelevant columns
df = df.drop(['Patient Number', 'Data Accuracy (%)'], axis=1, errors='ignore')

# One-hot encode categorical features
df = pd.get_dummies(df, columns=categorical_cols[:-1], drop_first=True)

# Encode target
le = LabelEncoder()
df['Predicted Disease'] = le.fit_transform(df['Predicted Disease'])
y = to_categorical(df['Predicted Disease'])

# Define features
X = df.drop('Predicted Disease', axis=1)

# Save feature names
joblib.dump(X.columns.tolist(), 'feature_names.pkl')
print("Feature names saved to 'feature_names.pkl':", X.columns.tolist())

# Scale numerical features
scaler = StandardScaler()
X[numerical_cols + ['BP_Difference', 'Heart_Rate_SpO2_Ratio']] = scaler.fit_transform(X[numerical_cols + ['BP_Difference', 'Heart_Rate_SpO2_Ratio']])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build neural network
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train model
print("Training Neural Network...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_classes, y_pred_classes)

print(f"\nNeural Network Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=le.classes_))

# Save model, scaler, and label encoder
model.save('disease_prediction_model.keras')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("\nModel, scaler, and label encoder saved successfully.")