import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Carrega o dataset
df = pd.read_csv("dataset/Dataset Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

df_model = df.drop(columns=['customerID']).copy()
df_model['Churn'] = (df_model['Churn'] == 'Yes').astype(int)
X_raw = df_model.drop(columns=['Churn'])
X = pd.get_dummies(X_raw, drop_first=True)

# Treina o scaler
scaler = StandardScaler()
scaler.fit(X)

# Salva
dest_dir = Path('models/saved')
dest_dir.mkdir(parents=True, exist_ok=True)
preprocessor_path = dest_dir / 'preprocessor.pkl'

with open(preprocessor_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"Preprocessador salvo em: {preprocessor_path}")
print(f"Tamanho: {preprocessor_path.stat().st_size} bytes")
