import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load dataset
data = pd.read_csv("india_2000_2024_daily_weather.csv")

# Clean data
data = data.dropna()

# Features
X = data[['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']]

# ================= FLOOD MODEL =================
y_flood = (data['precipitation_sum'] > 100).astype(int)

flood_model = RandomForestClassifier()
flood_model.fit(X, y_flood)

# ================= DROUGHT MODEL =================
y_drought = (data['precipitation_sum'] < 10).astype(int)

drought_model = RandomForestClassifier()
drought_model.fit(X, y_drought)

# ================= HEATWAVE MODEL =================
y_heatwave = (data['temperature_2m_max'] > 40).astype(int)

heatwave_model = RandomForestClassifier()
heatwave_model.fit(X, y_heatwave)

# ================= SAVE MODELS =================
os.makedirs("models", exist_ok=True)

with open("models/flood_model.pkl", "wb") as f:
    pickle.dump(flood_model, f)

with open("models/drought_model.pkl", "wb") as f:
    pickle.dump(drought_model, f)

with open("models/heatwave_model.pkl", "wb") as f:
    pickle.dump(heatwave_model, f)

print("✅ All models trained and saved successfully!")