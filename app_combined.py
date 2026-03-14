# ============================================================
# COMBINED FLASK APP - ML ONLY (NO RULE BASED FALLBACK)
# ============================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
import requests
from datetime import datetime

# Import ML prediction module
try:
    from flood_predictor import predict_disasters, load_models
    print("✅ Prediction module imported")
except ImportError as e:
    print("❌ Could not import flood_predictor:", e)
    sys.exit(1)

app = Flask(__name__, static_folder='.')
CORS(app)

# ================= TELEGRAM CONFIG =================

TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

# ================= LOAD ML MODELS =================

print("🚀 Checking ML models...")

try:
    flood_model, drought_model, heatwave_model = load_models()

    # If models are missing → stop server
    if flood_model is None or drought_model is None or heatwave_model is None:
        print("❌ ML models not found. Train models first.")
        sys.exit(1)

    print("✅ ML models loaded successfully")

except Exception as e:
    print("❌ Failed to load ML models:", e)
    sys.exit(1)

# ================= CITY COORDINATES =================

CITY_COORDINATES = {
    "mumbai": {"lat": 19.0760, "lon": 72.8777},
    "delhi": {"lat": 28.7041, "lon": 77.1025},
    "bangalore": {"lat": 12.9716, "lon": 77.5946},
    "chennai": {"lat": 13.0827, "lon": 80.2707},
    "kolkata": {"lat": 22.5726, "lon": 88.3639},
    "hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "pune": {"lat": 18.5204, "lon": 73.8567},
}

def get_coordinates(city):
    city = city.lower().strip()
    if city in CITY_COORDINATES:
        return CITY_COORDINATES[city]["lat"], CITY_COORDINATES[city]["lon"]
    return None, None


# ================= HOME PAGE =================

@app.route('/')
def index():
    return send_from_directory('.', 'index_combined.html')


# ================= ML PREDICTION =================

@app.route('/api/predict', methods=['POST'])
def predict():

    try:
        data = request.get_json()
        city = data.get("city")

        if not city:
            return jsonify({
                "success": False,
                "message": "City required"
            }), 400

        lat, lon = get_coordinates(city)

        if lat is None:
            return jsonify({
                "success": False,
                "message": "City not supported"
            }), 404

        # ML prediction only
        result = predict_disasters(city, lat, lon, use_ml=True)

        if result is None or result["method"] != "Machine Learning":
            return jsonify({
                "success": False,
                "message": "ML prediction unavailable"
            }), 500

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# ================= ALL CITY PREDICTIONS =================

@app.route('/api/predict-all-cities', methods=['POST'])
def predict_all():

    results = []

    for city, coords in CITY_COORDINATES.items():

        prediction = predict_disasters(
            city.title(),
            coords["lat"],
            coords["lon"],
            use_ml=True
        )

        # Skip if ML failed
        if prediction is None or prediction["method"] != "Machine Learning":
            continue

        probs = {
            k: v["probability_percent"]
            for k, v in prediction["predictions"].items()
        }

        max_risk = max(probs.values())
        main_type = max(probs, key=probs.get)

        results.append({
            "city": city.title(),
            "lat": coords["lat"],
            "lon": coords["lon"],
            "max_risk": max_risk,
            "disaster_type": main_type,
            "predictions": prediction["predictions"]
        })

    if len(results) == 0:
        return jsonify({
            "success": False,
            "message": "ML prediction failed for all cities"
        }), 500

    return jsonify({
        "success": True,
        "cities": results
    })


# ================= SOS =================

@app.route('/sos', methods=['POST'])
def sos():

    data = request.json

    try:
        lat = float(data.get("latitude"))
        lon = float(data.get("longitude"))
        name = data.get("name", "Unknown")

        message = f"""
🚨 SOS ALERT 🚨

Name: {name}
Location: {lat}, {lon}
Time: {datetime.now()}
"""

        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message}
        )

        return jsonify({"status": "sent"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ================= START SERVER =================

if __name__ == "__main__":

    print("\n==============================")
    print("🌐 Server starting")
    print("⚙ Mode: ML ONLY")
    print("==============================\n")

    app.run(host="0.0.0.0", port=5000, debug=True)