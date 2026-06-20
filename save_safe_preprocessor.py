import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.utils.config import DATASET_PATH, PREPROCESSOR_PATH

# Load dataset
df = pd.read_csv(DATASET_PATH)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

df_model = df.drop(columns=['customerID']).copy()
df_model['Churn'] = (df_model['Churn'] == 'Yes').astype(int)
X_raw = df_model.drop(columns=['Churn'])
X = pd.get_dummies(X_raw, drop_first=True)

scaler = StandardScaler()
scaler.fit(X)

safe = (list(X.columns), scaler)
PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(safe, PREPROCESSOR_PATH)
print('Saved safe preprocessor at', PREPROCESSOR_PATH)
print('Num features:', len(safe[0]))
