from pathlib import Path
import pandas as pd

from src.preprocessing.pipeline import fit_and_save_preprocessor
from src.utils.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, DATASET_PATH

print('Dataset path:', DATASET_PATH)

df = pd.read_csv(DATASET_PATH)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

X_raw = df.drop(columns=['customerID', 'Churn'])

print('Fitting preprocessor...')
pre = fit_and_save_preprocessor(X_raw, NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
print('Preprocessor trained and saved.')
print('Preprocessor type:', type(pre))
print('Saved to:', Path(pre.steps[0][1].__class__.__module__))
