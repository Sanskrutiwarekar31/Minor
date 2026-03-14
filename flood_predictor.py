import os
import requests
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, date, timedelta
from difflib import get_close_matches

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except:
    SMOTE_AVAILABLE = False

import joblib

WEATHER_API_KEY = "14c27bb50b0044d6b37175716263101"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAINFALL_NORMAL_PATH = os.path.join(BASE_DIR, "district wise rainfall normal.csv")
DAILY_WEATHER_PATH = os.path.join(BASE_DIR, "india_2000_2024_daily_weather.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

try:
    rainfall_df = pd.read_csv(RAINFALL_NORMAL_PATH)
    rainfall_df["DISTRICT"] = rainfall_df["DISTRICT"].str.upper().str.strip()
    rainfall_df.fillna(0, inplace=True)
except FileNotFoundError:
    rainfall_df = None

try:
    daily_df = pd.read_csv(DAILY_WEATHER_PATH)
    daily_df.columns = daily_df.columns.str.lower().str.strip()
    
    col_map = {}
    for col in daily_df.columns:
        col_lower = col.lower()
        if 'date' in col_lower: col_map['date'] = col
        elif 'city' in col_lower or 'location' in col_lower or 'station' in col_lower: col_map['city'] = col
        elif 'tmax' in col_lower or 'max_temp' in col_lower: col_map['tmax'] = col
        elif 'tmin' in col_lower or 'min_temp' in col_lower: col_map['tmin'] = col
        elif 'tmean' in col_lower or 'mean_temp' in col_lower or 'avg_temp' in col_lower or 'tavg' in col_lower: col_map['tmean'] = col
        elif 'rain' in col_lower or 'precip' in col_lower: col_map['rain'] = col
        elif 'wind' in col_lower: col_map['wind'] = col
        elif 'humid' in col_lower: col_map['humidity'] = col
    
    if col_map:
        daily_df = daily_df.rename(columns={v: k for k, v in col_map.items()})
    
    if 'date' in daily_df.columns:
        daily_df["date"] = pd.to_datetime(daily_df["date"], errors='coerce')
        daily_df = daily_df.dropna(subset=['date'])
        daily_df = daily_df.sort_values(["city", "date"]) if 'city' in daily_df.columns else daily_df.sort_values("date")
    
    if 'tmean' not in daily_df.columns and 'tmax' in daily_df.columns and 'tmin' in daily_df.columns:
        daily_df['tmean'] = (daily_df['tmax'] + daily_df['tmin']) / 2
    
    if 'wind' not in daily_df.columns:
        daily_df['wind'] = 10
    if 'humidity' not in daily_df.columns:
        daily_df['humidity'] = 70
    
    if 'date' in daily_df.columns and 'city' in daily_df.columns and 'rain' in daily_df.columns:
        daily_df['month'] = daily_df['date'].dt.month
        monthly_rain_patterns = daily_df.groupby(['city', 'month'])['rain'].mean().reset_index()
        monthly_rain_patterns.rename(columns={'rain': 'monthly_avg_rain'}, inplace=True)
    else:
        monthly_rain_patterns = None
    
except FileNotFoundError:
    daily_df = None
    monthly_rain_patterns = None
except Exception as e:
    daily_df = None
    monthly_rain_patterns = None

def get_today_weather(city):
    url = f"https://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        d = r.json()["current"]
        return {
            "rain_mm": d.get("precip_mm", 0),
            "humidity": d.get("humidity", 0),
            "temp_c": d.get("temp_c", 0),
            "wind_kph": d.get("wind_kph", 0),
            "pressure_mb": d.get("pressure_mb", 0),
            "cloud": d.get("cloud", 0)
        }
    except:
        return None

def get_past_weather(lat, lon, days=30):
    end = date.today()
    start = end - timedelta(days=days)
    
    try:
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start.strftime('%Y-%m-%d')}&end_date={end.strftime('%Y-%m-%d')}"
            "&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,wind_speed_10m_max"
            "&timezone=auto"
        )
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if "daily" in data:
                d = data["daily"]
                df = pd.DataFrame({
                    "tmax": d["temperature_2m_max"],
                    "tmin": d["temperature_2m_min"],
                    "tmean": d["temperature_2m_mean"],
                    "rain": d["precipitation_sum"],
                    "wind": d["wind_speed_10m_max"],
                    "humidity": [70] * len(d["temperature_2m_max"])
                })
                return df
    except:
        pass
    
    month = datetime.now().month
    if abs(lat) < 15:
        base_temp = 28 + 2 * np.sin((month - 4) * np.pi / 6)
    elif abs(lat) < 35:
        base_temp = 25 + 8 * np.sin((month - 4) * np.pi / 6)
    else:
        base_temp = 15 + 12 * np.sin((month - 4) * np.pi / 6)
    
    is_monsoon = month in [6,7,8,9] and 8 <= lat <= 30 and 65 <= lon <= 95
    
    df = pd.DataFrame({
        "tmax": base_temp + 5 + np.random.randn(days) * 2,
        "tmin": base_temp - 5 + np.random.randn(days) * 2,
        "tmean": base_temp + np.random.randn(days) * 1.5,
        "rain": np.random.exponential(10 if is_monsoon else 2, days),
        "wind": 10 + np.random.randn(days) * 3,
        "humidity": 60 + (20 if is_monsoon else 0) + np.random.randn(days) * 10
    })
    
    df["tmax"] = df["tmax"].clip(lower=df["tmin"] + 2)
    df["rain"] = df["rain"].clip(lower=0)
    df["wind"] = df["wind"].clip(lower=0)
    df["humidity"] = df["humidity"].clip(lower=20, upper=100)
    
    return df

def get_max_consecutive(boolean_series):
    max_count = current_count = 0
    for val in boolean_series:
        if val:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    return max_count

def calculate_advanced_features(past_df, today_weather=None, city=None):
    features = {}
    n = len(past_df)
    
    features["temp_mean_30d"] = past_df["tmean"].mean()
    features["temp_mean_15d"] = past_df["tmean"].tail(min(15,n)).mean()
    features["temp_mean_7d"]  = past_df["tmean"].tail(min(7,n)).mean()
    features["temp_mean_3d"]  = past_df["tmean"].tail(min(3,n)).mean()
    
    features["temp_max_30d"] = past_df["tmax"].max()
    features["temp_max_7d"]  = past_df["tmax"].tail(min(7,n)).max()
    features["temp_min_30d"] = past_df["tmin"].min()
    
    features["temp_std_30d"] = past_df["tmean"].std()
    features["temp_std_7d"]  = past_df["tmean"].tail(min(7,n)).std()
    
    if n >= 14:
        features["temp_trend_7d"] = past_df["tmean"].tail(7).mean() - past_df["tmean"].tail(14).head(7).mean()
    else:
        features["temp_trend_7d"] = 0
    
    if n >= 30:
        features["temp_trend_30d"] = past_df["tmean"].tail(15).mean() - past_df["tmean"].head(15).mean()
    else:
        features["temp_trend_30d"] = 0
    
    temp_95 = past_df["tmax"].quantile(0.95)
    features["extreme_heat_days"]   = (past_df["tmax"].tail(min(15,n)) > temp_95).sum()
    features["consecutive_hot_days"] = get_max_consecutive(past_df["tmax"].tail(min(30,n)) > past_df["tmax"].quantile(0.90))
    
    features["rain_sum_30d"] = past_df["rain"].sum()
    features["rain_sum_15d"] = past_df["rain"].tail(min(15,n)).sum()
    features["rain_sum_7d"]  = past_df["rain"].tail(min(7,n)).sum()
    features["rain_sum_3d"]  = past_df["rain"].tail(min(3,n)).sum()
    
    features["rain_max_30d"] = past_df["rain"].max()
    features["rain_max_7d"]  = past_df["rain"].tail(min(7,n)).max()
    features["rain_mean_30d"] = past_df["rain"].mean()
    features["rain_std_30d"]  = past_df["rain"].std()
    
    features["dry_days_30d"] = (past_df["rain"] == 0).sum()
    features["dry_days_15d"] = (past_df["rain"].tail(min(15,n)) == 0).sum()
    features["dry_days_7d"]  = (past_df["rain"].tail(min(7,n)) == 0).sum()
    
    features["max_consecutive_dry_days"] = get_max_consecutive(past_df["rain"] == 0)
    features["heavy_rain_days_30d"] = (past_df["rain"] > 50).sum()
    features["heavy_rain_days_7d"]  = (past_df["rain"].tail(min(7,n)) > 50).sum()
    
    rainy = past_df[past_df["rain"] > 0]["rain"]
    features["rain_intensity"] = rainy.mean() if len(rainy) > 0 else 0
    features["rain_anomaly_30d"] = past_df["rain"].tail(min(30,n)).sum() / (past_df["rain"].mean() * 30 + 0.01)
    
    features["humidity_mean_30d"] = past_df["humidity"].mean()
    features["humidity_mean_7d"]  = past_df["humidity"].tail(min(7,n)).mean()
    features["humidity_max_7d"]   = past_df["humidity"].tail(min(7,n)).max()
    features["high_humidity_days"] = (past_df["humidity"].tail(min(15,n)) > 80).sum()
    
    features["wind_mean_30d"] = past_df["wind"].mean()
    features["wind_max_30d"]  = past_df["wind"].max()
    features["wind_mean_7d"]  = past_df["wind"].tail(min(7,n)).mean()
    
    features["heat_index"]   = features["temp_mean_7d"] * 0.7 + features["humidity_mean_7d"] * 0.3
    features["drought_index"] = features["max_consecutive_dry_days"] * 0.4 + features["temp_mean_15d"] * 0.3 + (100 - features["humidity_mean_30d"]) * 0.3
    features["flood_index"]   = features["rain_sum_7d"] * 0.5 + features["heavy_rain_days_7d"] * 10 + features["humidity_mean_7d"] * 0.3
    
    if today_weather:
        features["today_rain"]     = today_weather.get("rain_mm", 0)
        features["today_temp"]     = today_weather.get("temp_c", 0)
        features["today_humidity"] = today_weather.get("humidity", 0)
        features["today_wind"]     = today_weather.get("wind_kph", 0)
        features["temp_change"]    = abs(today_weather.get("temp_c", 0) - features["temp_mean_3d"])
    
    month = datetime.now().month
    features["month"]       = month
    features["is_monsoon"]  = 1 if month in [6,7,8,9] else 0
    features["is_summer"]   = 1 if month in [3,4,5] else 0
    features["is_winter"]   = 1 if month in [11,12,1,2] else 0
    
    if monthly_rain_patterns is not None and city is not None:
        city_month_avg = monthly_rain_patterns[(monthly_rain_patterns['city'].str.lower() == city.lower()) & (monthly_rain_patterns['month'] == month)]['monthly_avg_rain']
        if not city_month_avg.empty:
            historical_avg = city_month_avg.iloc[0]
            features["rain_vs_historical"] = features["rain_sum_30d"] / (historical_avg + 0.01)
            features["rain_deficit"] = max(0, historical_avg - features["rain_sum_30d"])
        else:
            features["rain_vs_historical"] = 1.0
            features["rain_deficit"] = 0
    else:
        features["rain_vs_historical"] = 1.0
        features["rain_deficit"] = 0
    
    return features

def rule_based_prediction(features):
    flood = 0
    if features.get("rain_sum_7d", 0)   > 180: flood += 28
    elif features.get("rain_sum_7d", 0) > 120: flood += 22
    elif features.get("rain_sum_7d", 0) >  80: flood += 16
    elif features.get("rain_sum_7d", 0) >  40: flood += 9
    
    if features.get("rain_sum_3d", 0)   > 110: flood += 24
    elif features.get("rain_sum_3d", 0) >  70: flood += 18
    elif features.get("rain_sum_3d", 0) >  35: flood += 11
    
    if features.get("heavy_rain_days_7d", 0) >= 4: flood += 19
    elif features.get("heavy_rain_days_7d", 0) == 3: flood += 13
    elif features.get("heavy_rain_days_7d", 0) == 2: flood += 7
    
    if features.get("humidity_mean_7d", 0) > 92: flood += 14
    elif features.get("humidity_mean_7d", 0) > 85: flood += 10
    elif features.get("humidity_mean_7d", 0) > 78: flood += 5
    
    if features.get("today_rain", 0) > 55: flood += 18
    elif features.get("today_rain", 0) > 32: flood += 12
    elif features.get("today_rain", 0) > 15: flood += 6
    
    if features.get("rain_vs_historical", 1.0) > 1.5: flood += 8
    elif features.get("rain_vs_historical", 1.0) > 1.2: flood += 4
    
    flood = min(flood, 99)

    drought = 0
    if features.get("max_consecutive_dry_days", 0) > 22: drought += 26
    elif features.get("max_consecutive_dry_days", 0) > 16: drought += 20
    elif features.get("max_consecutive_dry_days", 0) > 11: drought += 14
    elif features.get("max_consecutive_dry_days", 0) >  7: drought += 8
    
    if features.get("rain_sum_30d", 0) < 12: drought += 23
    elif features.get("rain_sum_30d", 0) < 28: drought += 17
    elif features.get("rain_sum_30d", 0) < 55: drought += 10
    
    if features.get("temp_mean_15d", 0) > 38: drought += 19
    elif features.get("temp_mean_15d", 0) > 35: drought += 13
    elif features.get("temp_mean_15d", 0) > 32: drought += 7
    
    if features.get("humidity_mean_30d", 0) < 35: drought += 12
    elif features.get("humidity_mean_30d", 0) < 45: drought += 8
    
    if features.get("rain_vs_historical", 1.0) < 0.5: drought += 9
    elif features.get("rain_vs_historical", 1.0) < 0.8: drought += 5
    
    if features.get("rain_deficit", 0) > 50: drought += 15
    elif features.get("rain_deficit", 0) > 20: drought += 8
    
    drought = min(drought, 99)

    heat = 0
    if features.get("temp_max_7d", 0) > 44: heat += 27
    elif features.get("temp_max_7d", 0) > 41: heat += 21
    elif features.get("temp_max_7d", 0) > 38: heat += 15
    elif features.get("temp_max_7d", 0) > 35: heat += 8
    
    if features.get("temp_mean_7d", 0) > 39: heat += 22
    elif features.get("temp_mean_7d", 0) > 36: heat += 16
    elif features.get("temp_mean_7d", 0) > 33: heat += 9
    
    if features.get("consecutive_hot_days", 0) > 5: heat += 18
    elif features.get("consecutive_hot_days", 0) > 3: heat += 12
    elif features.get("consecutive_hot_days", 0) == 2: heat += 6
    
    if features.get("humidity_mean_7d", 0) < 28: heat += 11
    elif features.get("humidity_mean_7d", 0) < 35: heat += 7
    
    if features.get("today_temp", 0) > 42: heat += 16
    elif features.get("today_temp", 0) > 39: heat += 10
    
    heat = min(heat, 99)

    return flood, drought, heat

def prepare_training_data():
    if daily_df is None or len(daily_df) < 100:
        return None, None, None
    
    required_cols = ['tmax','tmin','tmean','rain']
    if any(col not in daily_df.columns for col in required_cols):
        return None, None, None
    
    X_flood, y_flood = [], []
    X_drought, y_drought = [], []
    X_heatwave, y_heatwave = [], []
    
    cities = daily_df["city"].unique() if 'city' in daily_df.columns else [None]
    
    sample_count = 0
    for city in cities:
        if city is not None:
            city_data = daily_df[daily_df["city"] == city].sort_values("date")
        else:
            city_data = daily_df.sort_values("date")
        
        if len(city_data) < 60:
            continue
        
        for i in range(60, len(city_data), 10):
            past = city_data.iloc[i-30:i]
            future = city_data.iloc[i:i+7]
            
            if len(future) < 7:
                continue
            
            try:
                feat = calculate_advanced_features(past, city=city)
                vec = list(feat.values())
                
                flood_label   = 1 if future["rain"].sum() > 100 else 0
                drought_label = 1 if ((future["rain"]==0).sum() >=5 and future["tmean"].mean()>35) else 0
                heat_label    = 1 if (future["tmax"].max()>42 and future["tmax"].mean()>38) else 0
                
                X_flood.append(vec);   y_flood.append(flood_label)
                X_drought.append(vec); y_drought.append(drought_label)
                X_heatwave.append(vec);y_heatwave.append(heat_label)
                
                sample_count += 1
            except:
                continue
        
        if sample_count >= 1000:
            break
    
    if len(X_flood) < 50:
        return None, None, None
    
    return (
        (np.array(X_flood),   np.array(y_flood)),
        (np.array(X_drought), np.array(y_drought)),
        (np.array(X_heatwave),np.array(y_heatwave))
    )

class DisasterPredictor:
    def __init__(self, disaster_type):
        self.disaster_type = disaster_type
        self.scaler = RobustScaler()
        self.model = None
        
    def train(self, X, y):
        if len(X) < 50:
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)
        
        self.model = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, max_features='sqrt', random_state=42,
            n_jobs=-1, class_weight='balanced'
        )
        self.model.fit(X_train_s, y_train)
        
        y_pred = self.model.predict(X_test_s)
        return {"acc": accuracy_score(y_test, y_pred), "f1": f1_score(y_test, y_pred)}
    
    def predict_proba(self, X):
        if self.model is None:
            return 50.0
        X_s = self.scaler.transform(X)
        return self.model.predict_proba(X_s)[:,1][0] * 100

def train_all_models():
    result = prepare_training_data()
    if result is None or result[0] is None:
        return None, None, None
    
    (Xf,yf), (Xd,yd), (Xh,yh) = result
    
    fp = DisasterPredictor("FLOOD")
    dp = DisasterPredictor("DROUGHT")
    hp = DisasterPredictor("HEATWAVE")
    
    fp.train(Xf,yf)
    dp.train(Xd,yd)
    hp.train(Xh,yh)
    
    try:
        joblib.dump(fp, os.path.join(MODEL_DIR, "flood_model.pkl"))
        joblib.dump(dp, os.path.join(MODEL_DIR, "drought_model.pkl"))
        joblib.dump(hp, os.path.join(MODEL_DIR, "heatwave_model.pkl"))
    except:
        pass
    
    return fp, dp, hp

def load_models():
    try:
        return (
            joblib.load(os.path.join(MODEL_DIR, "flood_model.pkl")),
            joblib.load(os.path.join(MODEL_DIR, "drought_model.pkl")),
            joblib.load(os.path.join(MODEL_DIR, "heatwave_model.pkl"))
        )
    except:
        return train_all_models()

def predict_disasters(city, lat, lon, use_ml=True):
    today = get_today_weather(city)
    if not today:
        return None
    
    past = get_past_weather(lat, lon, days=30)
    if past is None or len(past) == 0:
        return None
    
    features = calculate_advanced_features(past, today, city=city)
    
    if use_ml:
        models = load_models()
        if models[0] is not None and models[0].model is not None:
            fm, dm, hm = models
            fv = np.array([list(features.values())])
            fp = fm.predict_proba(fv)
            dp = dm.predict_proba(fv)
            hp = hm.predict_proba(fv)
            method = "Machine Learning"
        else:
            fp, dp, hp = rule_based_prediction(features)
            method = "Rule-Based"
    else:
        fp, dp, hp = rule_based_prediction(features)
        method = "Rule-Based"
    
    return {
        "location": city,
        "latitude": lat,
        "longitude": lon,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": method,
        "current_conditions": today,
        "predictions": {
            "flood":   {"probability_percent": round(fp, 2),   "risk_level": get_risk_level(fp)},
            "drought": {"probability_percent": round(dp, 2),   "risk_level": get_risk_level(dp)},
            "heatwave": {"probability_percent": round(hp, 2), "risk_level": get_risk_level(hp)}
        },
        "key_factors": get_key_factors(features, fp, dp, hp)
    }

def get_risk_level(p):
    if p < 15: return "Very Low"
    if p < 30: return "Low"
    if p < 50: return "Moderate"
    if p < 75: return "High"
    return "Very High"

def get_key_factors(f, fp, dp, hp):
    fac = []
    if fp > 35:
        if f.get("rain_sum_7d",0) > 70: fac.append("Heavy rainfall in past week")
        if f.get("humidity_mean_7d",0) > 80: fac.append("High humidity levels")
        if f.get("heavy_rain_days_7d",0) > 2: fac.append("Multiple heavy rain days")
    if dp > 35:
        if f.get("max_consecutive_dry_days",0) > 10: fac.append("Extended dry period")
        if f.get("temp_mean_15d",0) > 35: fac.append("Sustained high temperatures")
        if f.get("rain_sum_30d",0) < 20: fac.append("Very low monthly rainfall")
    if hp > 35:
        if f.get("temp_max_7d",0) > 40: fac.append("Extreme temperature recorded")
        if f.get("consecutive_hot_days",0) > 3: fac.append("Consecutive hot days")
        if f.get("humidity_mean_7d",0) < 30: fac.append("Very low humidity")
    return fac if fac else ["No significant risk factors detected"]

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_all_models()
    else:
        try:
            city = input("Enter city: ").strip()
            lat = float(input("Latitude: ").strip())
            lon = float(input("Longitude: ").strip())
            r = predict_disasters(city, lat, lon)
            if r:
                print("\n" + "="*60)
                print(f"Location: {r['location']}")
                print(f"Method: {r['method']}")
                print("-"*60)
                print("PREDICTIONS:")
                for d, v in r['predictions'].items():
                    print(f"  {d.upper()}: {v['probability_percent']}% → {v['risk_level']}")
                print("-"*60)
                print("KEY FACTORS:")
                for f in r['key_factors']:
                    print(f"  • {f}")
                print("="*60)
        except Exception as e:
            print(f"Error: {e}")
