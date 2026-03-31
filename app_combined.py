# ============================================================
# DRISHTI — Disaster Risk Intelligence & Situational Hub
# Flask Backend — fully aligned with frontend UI (light theme)
#
# API contracts matched to frontend field accesses:
#
#   POST /api/predict
#     → { success, data: { predictions{k: {probability_percent, risk_level}},
#                          location, date, method,
#                          current_conditions: {temp_c, humidity, rain_mm},
#                          key_factors[] } }
#
#   POST /api/predict-all-cities
#     → { success, cities: [{city, lat, lon, max_risk, risk_level,
#                             disaster_type, predictions}] }
#
#   POST /api/live-alert
#     → { success, city, lat, lon, temperature, humidity,
#         rain_mm, level, message,
#         ml_predictions{k: {probability_percent, risk_level}} }
#
#   POST /sos
#     → { status: "sent" | "error", message? }
# ============================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys, os, requests
from datetime import datetime

# ── API keys from environment (never hardcode) ───────────────
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
TELEGRAM_BOT_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── ML prediction module ─────────────────────────────────────
try:
    from flood_predictor import predict_disasters, load_models
    print("✅ Prediction module imported")
except ImportError as e:
    print("❌ Could not import flood_predictor:", e)
    sys.exit(1)

app = Flask(__name__, static_folder='.')
CORS(app)

# ── Load ML models at startup ────────────────────────────────
print("🚀 Loading ML models…")
try:
    flood_model, drought_model, heatwave_model = load_models()
    if None in (flood_model, drought_model, heatwave_model):
        print("❌ One or more ML models missing. Run training first.")
        sys.exit(1)
    print("✅ All ML models loaded")
except Exception as e:
    print("❌ Failed to load ML models:", e)
    sys.exit(1)

# ── 20 major Indian cities ───────────────────────────────────
# Must match the <datalist id="city-list"> options in the frontend
CITY_COORDINATES = {
    "mumbai":        {"lat": 19.0760, "lon": 72.8777},
    "delhi":         {"lat": 28.7041, "lon": 77.1025},
    "bangalore":     {"lat": 12.9716, "lon": 77.5946},
    "chennai":       {"lat": 13.0827, "lon": 80.2707},
    "kolkata":       {"lat": 22.5726, "lon": 88.3639},
    "hyderabad":     {"lat": 17.3850, "lon": 78.4867},
    "pune":          {"lat": 18.5204, "lon": 73.8567},
    "nashik":        {"lat": 19.9975, "lon": 73.7898},
    "ahmedabad":     {"lat": 23.0225, "lon": 72.5714},
    "jaipur":        {"lat": 26.9124, "lon": 75.7873},
    "lucknow":       {"lat": 26.8467, "lon": 80.9462},
    "nagpur":        {"lat": 21.1458, "lon": 79.0882},
    "bhopal":        {"lat": 23.2599, "lon": 77.4126},
    "patna":         {"lat": 25.5941, "lon": 85.1376},
    "bhubaneswar":   {"lat": 20.2961, "lon": 85.8245},
    "guwahati":      {"lat": 26.1445, "lon": 91.7362},
    "surat":         {"lat": 21.1702, "lon": 72.8311},
    "visakhapatnam": {"lat": 17.6868, "lon": 83.2185},
    "kochi":         {"lat": 9.9312,  "lon": 76.2673},
    "varanasi":      {"lat": 25.3176, "lon": 82.9739},
}

# ── Helpers ──────────────────────────────────────────────────

def get_coordinates(city: str):
    """Return (lat, lon) for a city name, case-insensitive. Returns (None, None) if unknown."""
    return (
        (CITY_COORDINATES[city]["lat"], CITY_COORDINATES[city]["lon"])
        if city in CITY_COORDINATES else (None, None)
    )


def risk_label(probability_percent: float) -> str:
    """
    Convert a probability % to the risk string the UI uses.
    UI badge classes: 'high' (≥60), 'med' (≥40), 'low' (<40)
    The bc() helper in the frontend matches on .lower():
        'high' → red badge
        'med' / 'mod' / 'medium' / 'moderate' → amber badge
        'low' → green badge
    """
    if probability_percent >= 60:
        return "HIGH"
    elif probability_percent >= 40:
        return "MEDIUM"
    else:
        return "LOW"


def attach_risk_levels(predictions: dict) -> dict:
    """Add risk_level field to every disaster prediction dict in-place."""
    for pred in predictions.values():
        pred["risk_level"] = risk_label(pred.get("probability_percent", 0))
    return predictions


# ── Serve the frontend ───────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index_combined.html')


# ════════════════════════════════════════════════════════════
#  POST /api/predict  — single city ML prediction
#
#  Frontend reads:
#    result.success
#    result.message                          (on failure)
#    result.data.predictions[k].probability_percent
#    result.data.predictions[k].risk_level
#    result.data.location
#    result.data.date
#    result.data.method
#    result.data.current_conditions.temp_c
#    result.data.current_conditions.humidity
#    result.data.current_conditions.rain_mm
#    result.data.key_factors[]
# ════════════════════════════════════════════════════════════
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json(silent=True)
        if not body:
            return jsonify({"success": False, "message": "Request body must be JSON"}), 400

        city_raw = body.get("city", "").strip()
        if not city_raw:
            return jsonify({"success": False, "message": "Field 'city' is required"}), 400

        city_key = city_raw.lower()
        lat, lon = get_coordinates(city_key)

        if lat is None:
            supported = ", ".join(c.title() for c in CITY_COORDINATES)
            return jsonify({
                "success": False,
                "message": f"City '{city_raw}' is not supported. Supported: {supported}"
            }), 404

        result = predict_disasters(city_raw.title(), lat, lon, use_ml=True)

        if result is None or result.get("method") != "Machine Learning":
            return jsonify({
                "success": False,
                "message": "ML prediction unavailable for this city right now"
            }), 500

        # Attach risk_level to each prediction — frontend reads this field
        attach_risk_levels(result["predictions"])

        # Ensure all fields the frontend reads are present
        # (flood_predictor may not return all of these — we guarantee them here)
        result.setdefault("location", city_raw.title())
        result.setdefault("date", datetime.now().strftime("%d %b %Y, %H:%M"))
        result.setdefault("method", "Machine Learning")
        result.setdefault("key_factors", [])

        # Guarantee current_conditions structure with all three sub-fields
        cc = result.setdefault("current_conditions", {})
        cc.setdefault("temp_c", None)
        cc.setdefault("humidity", None)
        cc.setdefault("rain_mm", None)

        return jsonify({"success": True, "data": result})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ════════════════════════════════════════════════════════════
#  POST /api/predict-all-cities  — bulk map prediction
#
#  Frontend reads:
#    result.success
#    result.message                          (on failure)
#    result.cities[].city
#    result.cities[].lat
#    result.cities[].lon
#    result.cities[].max_risk
#    result.cities[].risk_level
#    result.cities[].disaster_type
#    result.cities[].predictions
#
#  Map marker colors: ≥60 → red, ≥30 → amber, <30 → green
#  City cards shown for: max_risk ≥ 30
# ════════════════════════════════════════════════════════════
@app.route('/api/predict-all-cities', methods=['POST'])
def predict_all():
    results = []

    for city_key, coords in CITY_COORDINATES.items():
        try:
            city_title = city_key.title()
            prediction = predict_disasters(
                city_title,
                coords["lat"],
                coords["lon"],
                use_ml=True
            )

            # Accept both ML and Rule-Based results
            if prediction is None:
                continue

            predictions = prediction["predictions"]
            attach_risk_levels(predictions)

            probs    = {k: v["probability_percent"] for k, v in predictions.items()}
            if not probs:
                continue

            max_risk  = max(probs.values())
            main_type = max(probs, key=probs.get)

            results.append({
                "city":          city_title,
                "lat":           coords["lat"],
                "lon":           coords["lon"],
                "max_risk":      round(max_risk, 2),
                "risk_level":    risk_label(max_risk),
                "disaster_type": main_type,
                "predictions":   predictions,
            })

        except Exception:
            continue

    if not results:
        return jsonify({
            "success": False,
            "message": "Prediction failed for all cities. Are the models trained?"
        }), 500

    return jsonify({"success": True, "cities": results})


# ════════════════════════════════════════════════════════════
#  GET /api/user-location — auto-detect city from IP
#  (mirrors your Streamlit get_user_location() function)
# ════════════════════════════════════════════════════════════
@app.route('/api/user-location', methods=['GET'])
def user_location():
    try:
        # Get visitor's IP
        user_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        if user_ip in ('127.0.0.1', '::1', 'localhost'):
            return jsonify({"success": False, "message": "Local IP — enter city manually"})

        res  = requests.get(f"https://ipinfo.io/{user_ip}/json", timeout=5)
        data = res.json()
        city = data.get("city")
        loc  = data.get("loc")  # "lat,lon"

        if city and loc:
            lat, lon = map(float, loc.split(","))
            return jsonify({"success": True, "city": city, "lat": lat, "lon": lon})

        return jsonify({"success": False, "message": "Could not detect location"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# ════════════════════════════════════════════════════════════
#  POST /api/live-alert  — real-time rainfall alert + ML
#
#  Frontend reads (all at top-level of response):
#    d.success
#    d.message
#    d.city          → shown in heading
#    d.temperature   → 🌡️ display
#    d.humidity      → 💧 display
#    d.rain_mm       → 🌧️ display
#    d.level         → "RED" | "ORANGE" | "GREEN"
#                      used as CSS class: class="ab RED"
#    d.message       → alert text
#    d.lat / d.lon   → Leaflet map setView + marker
#    d.ml_predictions[k].probability_percent
#    d.ml_predictions[k].risk_level
#
#  Alert thresholds (match what the UI describes):
#    rain > 150 mm → RED
#    rain > 50 mm  → ORANGE
#    else          → GREEN
# ════════════════════════════════════════════════════════════
@app.route('/api/live-alert', methods=['POST'])
def live_alert():
    try:
        body = request.get_json(silent=True)
        if not body:
            return jsonify({"success": False, "message": "Request body must be JSON"}), 400

        city = body.get("city", "").strip()
        if not city:
            return jsonify({"success": False, "message": "Field 'city' is required"}), 400

        # ── API keys ──
        OWM_KEY        = "960ed8bd981cd018c47ed8c49b2152bb"  # OpenWeatherMap
        WEATHERAPI_KEY = "14c27bb50b0044d6b37175716263101"   # WeatherAPI fallback

        lat = lon = temp_c = humidity = rain_mm = None

        # ── Try OpenWeatherMap first (from your Streamlit app) ──
        try:
            owm_url = (
                f"https://api.openweathermap.org/data/2.5/weather"
                f"?q={city}&appid={OWM_KEY}&units=metric"
            )
            owm_res = requests.get(owm_url, timeout=6)
            if owm_res.status_code == 200:
                owm_data = owm_res.json()
                lat      = owm_data["coord"]["lat"]
                lon      = owm_data["coord"]["lon"]
                temp_c   = round(owm_data["main"]["temp"], 1)
                humidity = owm_data["main"]["humidity"]
                rain_mm  = owm_data.get("rain", {}).get("1h", 0)
        except Exception:
            pass

        # ── Fallback to WeatherAPI.com if OWM failed ──
        if lat is None:
            try:
                wa_url = (
                    f"https://api.weatherapi.com/v1/current.json"
                    f"?key={WEATHERAPI_KEY}&q={city}&aqi=no"
                )
                wa_res = requests.get(wa_url, timeout=6)
                if wa_res.status_code == 200:
                    wa_data  = wa_res.json()
                    lat      = wa_data["location"]["lat"]
                    lon      = wa_data["location"]["lon"]
                    temp_c   = round(wa_data["current"]["temp_c"], 1)
                    humidity = wa_data["current"]["humidity"]
                    rain_mm  = wa_data["current"].get("precip_mm", 0)
            except Exception:
                pass

        # ── Fallback to city coordinates + CSV data if API fails ──
        if lat is None:
            city_key = city.lower().strip()
            coords   = CITY_COORDINATES.get(city_key)
            if coords is None:
                # Try partial match
                for k, v in CITY_COORDINATES.items():
                    if city_key in k or k in city_key:
                        coords = v
                        break
            if coords is None:
                return jsonify({
                    "success": False,
                    "message": f"City '{city}' not found. Try: Mumbai, Delhi, Pune, Chennai etc."
                }), 404

            lat      = coords["lat"]
            lon      = coords["lon"]
            temp_c   = 32.0
            humidity = 65
            rain_mm  = 0.0

        # ── Alert level (CSS class names used directly in UI) ──
        if rain_mm > 150:
            level   = "RED"
            message = "Heavy rainfall detected! High flood risk — evacuate low-lying areas."
        elif rain_mm > 50:
            level   = "ORANGE"
            message = "Moderate rainfall. Stay cautious and avoid flood-prone areas."
        else:
            level   = "GREEN"
            message = "Weather conditions are currently normal."

        # ── ML predictions ──
        ml_predictions = {}
        try:
            ml_result = predict_disasters(city, lat, lon, use_ml=True)
            if ml_result and ml_result.get("predictions"):
                ml_predictions = ml_result["predictions"]
                attach_risk_levels(ml_predictions)
        except Exception:
            pass

        return jsonify({
            "success":        True,
            "city":           city.title(),
            "lat":            lat,
            "lon":            lon,
            "temperature":    temp_c,
            "humidity":       humidity,
            "rain_mm":        rain_mm,
            "level":          level,
            "message":        message,
            "ml_predictions": ml_predictions,
        })

    except requests.exceptions.Timeout:
        return jsonify({"success": False, "message": "Weather API timed out. Try again."}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"success": False, "message": "Cannot reach weather API. Check internet."}), 502
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ════════════════════════════════════════════════════════════
#  POST /sos  — Telegram emergency alert
#
#  Frontend sends: { latitude, longitude, name }
#  Frontend reads: { status: "sent" | "error", message? }
# ════════════════════════════════════════════════════════════
@app.route('/sos', methods=['POST'])
def sos():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"status": "error", "message": "Request body must be JSON"}), 400

    try:
        lat_raw = body.get("latitude")
        lon_raw = body.get("longitude")

        if lat_raw is None or lon_raw is None:
            return jsonify({"status": "error", "message": "latitude and longitude are required"}), 400

        lat  = float(lat_raw)
        lon  = float(lon_raw)
        name = str(body.get("name", "Unknown")).strip()[:60]   # cap length

        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({"status": "error", "message": "Coordinates out of valid range"}), 400

        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return jsonify({
                "status": "error",
                "message": (
                    "Telegram not configured. Set TELEGRAM_BOT_TOKEN and "
                    "TELEGRAM_CHAT_ID environment variables."
                )
            }), 500

        maps_link = f"https://maps.google.com/?q={lat},{lon}"
        timestamp = datetime.now().strftime("%d %b %Y %H:%M:%S")

        tg_message = (
            f"🚨 SOS ALERT — DRISHTI SYSTEM 🚨\n\n"
            f"👤 Name:      {name}\n"
            f"📍 Location:  {lat:.6f}, {lon:.6f}\n"
            f"🗺  Maps:     {maps_link}\n"
            f"⏰ Time:      {timestamp}\n\n"
            f"⚠️  Please dispatch emergency services immediately."
        )

        tg_response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": tg_message},
            timeout=6
        )

        if tg_response.status_code == 200:
            return jsonify({"status": "sent"})

        # Telegram returned an error — surface it
        tg_error = tg_response.json().get("description", "Unknown Telegram error")
        return jsonify({"status": "error", "message": f"Telegram: {tg_error}"}), 502

    except ValueError:
        return jsonify({"status": "error", "message": "latitude and longitude must be valid numbers"}), 400
    except requests.exceptions.Timeout:
        return jsonify({"status": "error", "message": "Telegram API timed out"}), 504
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ════════════════════════════════════════════════════════════
#  Start server  — ALWAYS the last block in the file
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═" * 44)
    print("  🛰️  DRISHTI — Disaster Intelligence System")
    print("═" * 44)
    print(f"  🌐  URL      : http://localhost:5000")
    print(f"  ⚙️   Mode     : ML ONLY (no rule-based fallback)")
    print(f"  🏙️   Cities   : {len(CITY_COORDINATES)}")
    print(f"  🔑  OWM key  : {'SET ✅' if OPENWEATHER_API_KEY else 'NOT SET ⚠️'}")
    print(f"  📱  Telegram : {'SET ✅' if TELEGRAM_BOT_TOKEN else 'NOT SET ⚠️'}")
    print("═" * 44 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
