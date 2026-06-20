import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

from src.utils.config import DATASET_PATH, PREPROCESSOR_PATH

@dataclass
class LegacyPreprocessor:
    columns: list
    scaler: StandardScaler

    def transform(self, df):
        X = pd.get_dummies(df, drop_first=True)
        # Ensure same column order and fill missing with 0
        X = X.reindex(columns=self.columns, fill_value=0)
        return self.scaler.transform(X)

# Load dataset
print('Loading dataset...')
df = pd.read_csv(DATASET_PATH)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df_model = df.drop(columns=['customerID']).copy()
df_model['Churn'] = (df_model['Churn'] == 'Yes').astype(int)
X_raw = df_model.drop(columns=['Churn'])
X = pd.get_dummies(X_raw, drop_first=True)

print('Fitting scaler on get_dummies output...')
scaler = StandardScaler()
scaler.fit(X)

legacy = LegacyPreprocessor(columns=list(X.columns), scaler=scaler)
PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(legacy, PREPROCESSOR_PATH)
print('Legacy preprocessor saved to', PREPROCESSOR_PATH)
print('Num features:', len(legacy.columns))
