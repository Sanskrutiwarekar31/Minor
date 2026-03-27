# ============================================================
# flood_predictor.py — DRISHTI Disaster Prediction Engine
# Matched to: india_2000_2024_daily_weather.csv
# Columns: city, date, temperature_2m_max, temperature_2m_min,
#          apparent_temperature_max, apparent_temperature_min,
#          precipitation_sum, rain_sum, weather_code,
#          wind_speed_10m_max, wind_gusts_10m_max,
#          wind_direction_10m_dominant
# ============================================================

import os, warnings
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime, date, timedelta

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ── Paths ────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FLOOD_MODEL_PATH    = os.path.join(MODEL_DIR, "flood_model.pkl")
DROUGHT_MODEL_PATH  = os.path.join(MODEL_DIR, "drought_model.pkl")
HEATWAVE_MODEL_PATH = os.path.join(MODEL_DIR, "heatwave_model.pkl")
DAILY_WEATHER_PATH  = os.path.join(BASE_DIR,  "india_2000_2024_daily_weather.csv")

WEATHER_API_KEY = "14c27bb50b0044d6b37175716263101"

# ── Load historical data once at import ──────────────────────
print("📂 Loading historical weather data…")
try:
    _raw = pd.read_csv(DAILY_WEATHER_PATH)
    # Rename to internal short names
    _raw = _raw.rename(columns={
        "temperature_2m_max":          "tmax",
        "temperature_2m_min":          "tmin",
        "apparent_temperature_max":    "feels_max",
        "apparent_temperature_min":    "feels_min",
        "precipitation_sum":           "precip",
        "rain_sum":                    "rain",
        "wind_speed_10m_max":          "wind",
        "wind_gusts_10m_max":          "gusts",
        "wind_direction_10m_dominant": "wind_dir",
        "weather_code":                "wcode",
    })
    _raw["date"]  = pd.to_datetime(_raw["date"], errors="coerce")
    _raw["tmean"] = (_raw["tmax"] + _raw["tmin"]) / 2
    _raw["city"]  = _raw["city"].str.strip().str.title()
    _raw = _raw.dropna(subset=["date", "tmax", "tmin", "rain"])
    _raw = _raw.sort_values(["city", "date"]).reset_index(drop=True)
    DAILY_DF = _raw
    print(f"✅ Loaded {len(DAILY_DF):,} rows | Cities: {sorted(DAILY_DF['city'].unique())}")
except Exception as e:
    DAILY_DF = None
    print(f"⚠️  Could not load daily weather CSV: {e}")


# ══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (45 features — consistent every call)
# ══════════════════════════════════════════════════════════════
FEATURE_NAMES = [
    # Temperature
    "temp_mean_30d","temp_mean_15d","temp_mean_7d","temp_mean_3d",
    "temp_max_30d","temp_max_7d","temp_min_30d",
    "temp_std_30d","temp_std_7d",
    "temp_trend_7d","temp_trend_30d",
    "extreme_heat_days","consecutive_hot_days",
    # Rainfall
    "rain_sum_30d","rain_sum_15d","rain_sum_7d","rain_sum_3d",
    "rain_max_30d","rain_max_7d",
    "rain_mean_30d","rain_std_30d",
    "dry_days_30d","dry_days_15d","dry_days_7d",
    "max_consecutive_dry","heavy_rain_days_30d","heavy_rain_days_7d",
    "rain_intensity","precip_sum_30d","precip_sum_7d",
    # Humidity proxy (feels_max - tmax)
    "feels_diff_mean_7d","feels_diff_mean_30d",
    # Wind
    "wind_mean_30d","wind_max_30d","wind_mean_7d","gusts_max_30d",
    # Composite indices
    "heat_index","drought_index","flood_index",
    # Season
    "month","is_monsoon","is_summer","is_winter",
    # Today's live weather
    "today_rain","today_temp","today_wind",
]

assert len(FEATURE_NAMES) == 46, f"Expected 46 features, got {len(FEATURE_NAMES)}"


def _max_consec_true(series):
    """Max consecutive True values in a boolean series."""
    best = cur = 0
    for v in series:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def build_features(past: pd.DataFrame,
                   today_rain=0.0,
                   today_temp=30.0,
                   today_wind=10.0) -> np.ndarray:
    """
    Build exactly 45 features from a DataFrame of past 30 days.
    past must have columns: tmax, tmin, tmean, rain, precip,
                            wind, gusts, feels_max
    Returns shape (45,) numpy array.
    """
    n   = len(past)
    t7  = past["tmean"].tail(min(7,  n))
    t15 = past["tmean"].tail(min(15, n))
    t30 = past["tmean"]
    r7  = past["rain"].tail(min(7,  n))
    r15 = past["rain"].tail(min(15, n))
    r30 = past["rain"]
    w7  = past["wind"].tail(min(7,  n))
    w30 = past["wind"]

    # feels diff (heat-index proxy)
    fd7  = (past["feels_max"] - past["tmax"]).tail(min(7,  n)).mean()
    fd30 = (past["feels_max"] - past["tmax"]).mean()

    # temperature
    tm30 = t30.mean();    tm15 = t15.mean()
    tm7  = t7.mean();     tm3  = past["tmean"].tail(min(3,n)).mean()
    tmx30= past["tmax"].max()
    tmx7 = past["tmax"].tail(min(7,n)).max()
    tmn30= past["tmin"].min()
    ts30 = t30.std(ddof=0) if n>1 else 0
    ts7  = t7.std(ddof=0)  if len(t7)>1 else 0

    if n >= 14:
        tr7 = past["tmean"].tail(7).mean() - past["tmean"].tail(14).head(7).mean()
    else:
        tr7 = 0
    tr30 = past["tmean"].tail(15).mean() - past["tmean"].head(max(1,n-15)).mean() if n>=30 else 0

    q95      = past["tmax"].quantile(0.95)
    q90      = past["tmax"].quantile(0.90)
    exheat   = int((past["tmax"].tail(min(15,n)) > q95).sum())
    consec_h = _max_consec_true(past["tmax"].tail(min(30,n)) > q90)

    # rainfall
    rs30 = r30.sum(); rs15 = r15.sum(); rs7 = r7.sum()
    rs3  = past["rain"].tail(min(3,n)).sum()
    rmx30= r30.max(); rmx7 = r7.max()
    rmn30= r30.mean(); rstd30= r30.std(ddof=0) if n>1 else 0
    dry30= int((r30 == 0).sum())
    dry15= int((r15 == 0).sum())
    dry7 = int((r7  == 0).sum())
    mcd  = _max_consec_true(r30 == 0)
    hr30 = int((r30 > 50).sum())
    hr7  = int((r7  > 50).sum())
    rainy= r30[r30 > 0]
    ri   = rainy.mean() if len(rainy) > 0 else 0
    ps30 = past["precip"].sum() if "precip" in past.columns else rs30
    ps7  = past["precip"].tail(min(7,n)).sum() if "precip" in past.columns else rs7

    # wind
    wm30 = w30.mean(); wmx30= past["wind"].max()
    wm7  = w7.mean()
    gm30 = past["gusts"].max() if "gusts" in past.columns else wmx30

    # composite
    hi  = tm7  * 0.7 + (100 - fd7) * 0.3          # heat index proxy
    di  = mcd  * 0.4 + tm15 * 0.3 + (100-rmn30*2) * 0.3
    fi  = rs7  * 0.5 + hr7  * 10  + fd7 * 0.2

    # season
    month     = datetime.now().month
    is_monsoon= 1 if month in [6,7,8,9]    else 0
    is_summer = 1 if month in [3,4,5]      else 0
    is_winter = 1 if month in [11,12,1,2]  else 0

    vec = np.array([
        tm30, tm15, tm7, tm3,
        tmx30, tmx7, tmn30,
        ts30, ts7,
        tr7, tr30,
        exheat, consec_h,
        rs30, rs15, rs7, rs3,
        rmx30, rmx7,
        rmn30, rstd30,
        dry30, dry15, dry7,
        mcd, hr30, hr7,
        ri, ps30, ps7,
        fd7, fd30,
        wm30, wmx30, wm7, gm30,
        hi, di, fi,
        month, is_monsoon, is_summer, is_winter,
        float(today_rain), float(today_temp), float(today_wind),
    ], dtype=np.float64)

    # Replace NaN/Inf with 0
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    assert len(vec) == 46, f"Feature vector length {len(vec)} != 46"
    return vec


# ══════════════════════════════════════════════════════════════
#  DisasterPredictor wrapper
# ══════════════════════════════════════════════════════════════
class DisasterPredictor:
    def __init__(self, disaster_type: str):
        self.disaster_type = disaster_type
        self.scaler        = RobustScaler()
        self.model         = None
        self.n_features    = 46

    def train(self, X: np.ndarray, y: np.ndarray):
        if len(X) < 50:
            print(f"  ⚠️  Not enough samples for {self.disaster_type}: {len(X)}")
            return None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if y.sum() > 5 else None
        )
        X_tr = self.scaler.fit_transform(X_tr)
        X_te = self.scaler.transform(X_te)

        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=12,
            min_samples_split=8, min_samples_leaf=4,
            max_features="sqrt", random_state=42,
            n_jobs=-1, class_weight="balanced"
        )
        self.model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, self.model.predict(X_te))
        f1  = f1_score(y_te, self.model.predict(X_te), zero_division=0)
        print(f"  ✅ {self.disaster_type}: acc={acc:.3f}  f1={f1:.3f}  samples={len(X)}")
        return {"acc": acc, "f1": f1}

    def predict_proba(self, x: np.ndarray) -> float:
        """x shape: (45,) or (1,45). Returns probability 0-100."""
        if self.model is None:
            return 0.0
        x2 = x.reshape(1, -1)
        x2 = self.scaler.transform(x2)
        return float(self.model.predict_proba(x2)[0, 1]) * 100.0


# ══════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════
def _prepare_training_data():
    if DAILY_DF is None or len(DAILY_DF) < 200:
        return None

    # Need feels_max — if missing, approximate
    if "feels_max" not in DAILY_DF.columns:
        DAILY_DF["feels_max"] = DAILY_DF["tmax"] - 2

    Xf, yf = [], []
    Xd, yd = [], []
    Xh, yh = [], []

    cities = DAILY_DF["city"].unique()
    total  = 0

    for city in cities:
        cdf = DAILY_DF[DAILY_DF["city"] == city].sort_values("date").reset_index(drop=True)
        if len(cdf) < 60:
            continue

        # Step every 7 days for speed
        for i in range(30, len(cdf) - 7, 7):
            past   = cdf.iloc[i-30 : i]
            future = cdf.iloc[i    : i+7]
            if len(past) < 20 or len(future) < 7:
                continue
            try:
                vec = build_features(past)
            except Exception:
                continue

            fl = 1 if future["rain"].sum() > 50  else 0
            dr = 1 if ((future["rain"] == 0).sum() >= 4 and future["tmean"].mean() > 28) else 0
            hw = 1 if (future["tmax"].max() > 38 and future["tmax"].mean() > 34) else 0
           
            Xf.append(vec); yf.append(fl)
            Xd.append(vec); yd.append(dr)
            Xh.append(vec); yh.append(hw)
            total += 1

    print(f"  📊 Training samples: {total}")
    if total < 50:
        return None

    return (
        (np.array(Xf), np.array(yf)),
        (np.array(Xd), np.array(yd)),
        (np.array(Xh), np.array(yh)),
    )


def train_all_models():
    print("🚀 Training ML models…")
    data = _prepare_training_data()
    if data is None:
        print("❌ Not enough training data")
        return None, None, None

    (Xf,yf),(Xd,yd),(Xh,yh) = data

    fp = DisasterPredictor("FLOOD")
    dp = DisasterPredictor("DROUGHT")
    hp = DisasterPredictor("HEATWAVE")

    fp.train(Xf, yf)
    dp.train(Xd, yd)
    hp.train(Xh, yh)

    joblib.dump(fp, FLOOD_MODEL_PATH)
    joblib.dump(dp, DROUGHT_MODEL_PATH)
    joblib.dump(hp, HEATWAVE_MODEL_PATH)
    print("✅ Models saved to models/")
    return fp, dp, hp


# ══════════════════════════════════════════════════════════════
#  LOAD MODELS  (auto-retrain if stale / incompatible)
# ══════════════════════════════════════════════════════════════
def load_models():
    try:
        fm = joblib.load(FLOOD_MODEL_PATH)
        dm = joblib.load(DROUGHT_MODEL_PATH)
        hm = joblib.load(HEATWAVE_MODEL_PATH)

        # Validate they are DisasterPredictor with correct feature count
        for m in (fm, dm, hm):
            if not isinstance(m, DisasterPredictor):
                raise TypeError("Old model format — retraining")
            if m.n_features != 46:
                raise ValueError("Feature count mismatch — retraining")

        print("✅ Loaded existing models")
        return fm, dm, hm

    except Exception as e:
        print(f"⚠️  {e}")
        print("🔄 Retraining models…")
        return train_all_models()


# ══════════════════════════════════════════════════════════════
#  LIVE WEATHER  (WeatherAPI)
# ══════════════════════════════════════════════════════════════
def get_today_weather(city: str) -> dict:
    try:
        url = (
            f"https://api.weatherapi.com/v1/current.json"
            f"?key={WEATHER_API_KEY}&q={city}&aqi=no"
        )
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return None
        c = r.json()["current"]
        return {
            "rain_mm":     c.get("precip_mm",   0),
            "humidity":    c.get("humidity",    70),
            "temp_c":      c.get("temp_c",      30),
            "wind_kph":    c.get("wind_kph",    10),
            "pressure_mb": c.get("pressure_mb",1013),
            "cloud":       c.get("cloud",       50),
        }
    except Exception:
        return None


def get_past_weather_from_csv(city: str, days: int = 30) -> pd.DataFrame | None:
    """Pull last `days` rows for this city from the historical CSV."""
    if DAILY_DF is None:
        return None
    cdf = DAILY_DF[DAILY_DF["city"].str.lower() == city.lower()]
    if len(cdf) < days:
        # Try fuzzy match
        cities = DAILY_DF["city"].unique()
        match  = [c for c in cities if city.lower() in c.lower()]
        if match:
            cdf = DAILY_DF[DAILY_DF["city"] == match[0]]
    if len(cdf) == 0:
        return None
    return cdf.tail(days).reset_index(drop=True)


def get_past_weather_api(lat: float, lon: float, days: int = 30) -> pd.DataFrame:
    """Fetch recent weather from Open-Meteo archive API."""
    end   = date.today()
    start = end - timedelta(days=days)
    try:
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start}&end_date={end}"
            "&daily=temperature_2m_max,temperature_2m_min,"
            "temperature_2m_mean,precipitation_sum,rain_sum,"
            "wind_speed_10m_max,wind_gusts_10m_max"
            "&timezone=auto"
        )
        r = requests.get(url, timeout=12)
        if r.status_code == 200:
            d = r.json().get("daily", {})
            df = pd.DataFrame({
                "tmax":     d["temperature_2m_max"],
                "tmin":     d["temperature_2m_min"],
                "tmean":    d["temperature_2m_mean"],
                "precip":   d["precipitation_sum"],
                "rain":     d["rain_sum"],
                "wind":     d["wind_speed_10m_max"],
                "gusts":    d["wind_gusts_10m_max"],
                "feels_max":[ t-2 for t in d["temperature_2m_max"]],
            })
            df = df.fillna(0)
            return df
    except Exception:
        pass
    return None


def _fallback_past(lat: float, lon: float, days: int = 30) -> pd.DataFrame:
    """Synthetic fallback when all APIs fail."""
    month      = datetime.now().month
    base_temp  = 25 + 8 * np.sin((month - 4) * np.pi / 6)
    is_monsoon = month in [6,7,8,9] and 8 <= lat <= 30
    rain_scale = 12 if is_monsoon else 2
    tmax = base_temp + 5 + np.random.randn(days) * 2
    tmin = base_temp - 5 + np.random.randn(days) * 2
    rain = np.random.exponential(rain_scale, days).clip(0)
    return pd.DataFrame({
        "tmax":     tmax,
        "tmin":     tmin,
        "tmean":    (tmax+tmin)/2,
        "precip":   rain,
        "rain":     rain,
        "wind":     (10 + np.random.randn(days)*3).clip(0),
        "gusts":    (15 + np.random.randn(days)*4).clip(0),
        "feels_max":tmax - 2,
    })


# ══════════════════════════════════════════════════════════════
#  RISK HELPERS
# ══════════════════════════════════════════════════════════════
def _risk_level(p: float) -> str:
    if p < 20: return "LOW"
    if p < 45: return "MEDIUM"
    if p < 70: return "HIGH"
    return "VERY HIGH"


def _key_factors(feat_vec: np.ndarray, fp, dp, hp) -> list:
    # Map feature names to values
    f = dict(zip(FEATURE_NAMES, feat_vec))
    factors = []
    if fp > 35:
        if f.get("rain_sum_7d",0) > 70:     factors.append("Heavy rainfall in past week")
        if f.get("heavy_rain_days_7d",0)>2:  factors.append("Multiple heavy rain days recorded")
        if f.get("today_rain",0) > 20:       factors.append("Active rainfall today")
    if dp > 35:
        if f.get("max_consecutive_dry",0)>10: factors.append("Extended dry spell detected")
        if f.get("temp_mean_15d",0) > 35:    factors.append("Sustained high temperatures")
        if f.get("rain_sum_30d",0) < 20:     factors.append("Very low monthly rainfall")
    if hp > 35:
        if f.get("temp_max_7d",0) > 40:      factors.append("Extreme temperatures recorded")
        if f.get("consecutive_hot_days",0)>3: factors.append("Consecutive hot days")
        if f.get("today_temp",0) > 38:       factors.append("Severe heat today")
    return factors if factors else ["No significant risk factors detected"]


# ══════════════════════════════════════════════════════════════
#  MAIN PREDICTION FUNCTION  (called by Flask)
# ══════════════════════════════════════════════════════════════
def predict_disasters(city: str, lat: float, lon: float,
                      use_ml: bool = True) -> dict | None:

    # 1. Past 30 days from CSV (primary source — always available)
    past = get_past_weather_from_csv(city, days=31)
    if past is None or len(past) < 10:
        past = get_past_weather_api(lat, lon, days=30)
    if past is None or len(past) < 10:
        past = _fallback_past(lat, lon, days=30)

    # Ensure required columns exist
    if "feels_max" not in past.columns:
        past["feels_max"] = past["tmax"] - 2
    if "precip" not in past.columns:
        past["precip"] = past["rain"]
    if "gusts" not in past.columns:
        past["gusts"] = past["wind"] * 1.3

    # 2. Use last row of CSV as "today" — always realistic values
    last_row   = past.iloc[-1]
    today_rain = float(last_row.get("rain",   last_row.get("precip", 0)))
    today_temp = float(last_row.get("tmax",   30))
    today_wind = float(last_row.get("wind",   10))
    today_humidity = 70  # not in CSV — use default

    # Try live weather API on top (overrides CSV if successful)
    live = get_today_weather(city)
    if live:
        today_rain     = live.get("rain_mm",  today_rain)
        today_temp     = live.get("temp_c",   today_temp)
        today_wind     = live.get("wind_kph", today_wind)
        today_humidity = live.get("humidity", 70)

    # Use past 30 rows (exclude last row used as "today")
    past = past.iloc[:-1] if len(past) > 10 else past

    # Ensure required columns exist
    if "feels_max" not in past.columns:
        past["feels_max"] = past["tmax"] - 2
    if "precip" not in past.columns:
        past["precip"] = past["rain"]
    if "gusts" not in past.columns:
        past["gusts"] = past["wind"] * 1.3

    # 3. Build feature vector
    feat_vec = build_features(past, today_rain, today_temp, today_wind)

    # 4. Predict
    if use_ml:
        models = load_models()
        fm, dm, hm = models

        if (fm is not None and isinstance(fm, DisasterPredictor) and fm.model is not None):
            fp = fm.predict_proba(feat_vec)
            dp = dm.predict_proba(feat_vec)
            hp = hm.predict_proba(feat_vec)
            method = "Machine Learning"
        else:
            # Fallback to rule-based
            fp, dp, hp = _rule_based(feat_vec)
            method = "Rule-Based"
    else:
        fp, dp, hp = _rule_based(feat_vec)
        method = "Rule-Based"

    fp = round(float(np.clip(fp, 0, 99)), 1)
    dp = round(float(np.clip(dp, 0, 99)), 1)
    hp = round(float(np.clip(hp, 0, 99)), 1)

    # 5. current_conditions — exactly what the UI reads
    current_conditions = {
        "temp_c":   round(float(today_temp), 1),
        "humidity": int(today_humidity),
        "rain_mm":  round(float(today_rain), 1),
    }

    return {
        "location":           city.title(),
        "date":               datetime.now().strftime("%d %b %Y, %H:%M"),
        "method":             method,
        "current_conditions": current_conditions,
        "predictions": {
            "flood":    {"probability_percent": fp, "risk_level": _risk_level(fp)},
            "drought":  {"probability_percent": dp, "risk_level": _risk_level(dp)},
            "heatwave": {"probability_percent": hp, "risk_level": _risk_level(hp)},
        },
        "key_factors": _key_factors(feat_vec, fp, dp, hp),
    }


def _rule_based(feat_vec: np.ndarray):
    """Simple rule-based fallback using the feature vector."""
    f = dict(zip(FEATURE_NAMES, feat_vec))
    flood = min(99, (
        min(f.get("rain_sum_7d",0)/2, 40) +
        f.get("heavy_rain_days_7d",0)*8 +
        min(f.get("today_rain",0)*0.5, 20)
    ))
    drought = min(99, (
        f.get("max_consecutive_dry",0)*2 +
        max(0, f.get("temp_mean_15d",0)-30)*2 +
        max(0, 30-f.get("rain_sum_30d",30))
    ))
    heat = min(99, (
        max(0, f.get("temp_max_7d",0)-35)*5 +
        f.get("consecutive_hot_days",0)*6 +
        max(0, f.get("today_temp",0)-35)*3
    ))
    return flood, drought, heat


# ══════════════════════════════════════════════════════════════
#  CLI — python flood_predictor.py train
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_all_models()
    else:
        city = input("City: ").strip()
        lat  = float(input("Lat:  ").strip())
        lon  = float(input("Lon:  ").strip())
        r = predict_disasters(city, lat, lon)
        if r:
            print(f"\n{'='*50}")
            print(f"Location : {r['location']}")
            print(f"Method   : {r['method']}")
            print(f"Date     : {r['date']}")
            print(f"{'─'*50}")
            for k,v in r["predictions"].items():
                print(f"  {k.upper():10s}: {v['probability_percent']:5.1f}%  [{v['risk_level']}]")
            print(f"{'─'*50}")
            for fac in r["key_factors"]:
                print(f"  • {fac}")
            print(f"{'='*50}")
        else:
            print("Prediction failed.")
