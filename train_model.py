import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("district wise rainfall normal.csv")

print(df.columns)

# Create simple flood condition
df['Flood'] = df['ANNUAL'] > 1000

X = df[['ANNUAL']]
y = df['Flood']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "models/flood_model.pkl")

print("✅ Model trained and saved!")