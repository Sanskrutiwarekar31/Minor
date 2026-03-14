
# ============================================================
# COMBINED FLASK APP - DISASTER MANAGEMENT + SOS EMERGENCY
# ============================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import requests
from datetime import datetime

# Import prediction logic (your file is named flood_predictor.py)
try:
    from flood_predictor import predict_disasters, load_models, train_all_models
    print("✅ Imported prediction module from flood_predictor.py")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

app = Flask(__name__, static_folder='.')
CORS(app)

# Configuration
WEATHER_API_KEY = "14c27bb50b0044d6b37175716263101"
TELEGRAM_BOT_TOKEN = "8288138279:AAFg3Ql52TFRK-paXr_BCxxBEtu--fDaBYY"
TELEGRAM_CHAT_ID = 8581771628

# Load models (will fall back gracefully if training data is insufficient)
flood_model, drought_model, heatwave_model = None, None, None

def initialize_models():
    global flood_model, drought_model, heatwave_model
    print("🚀 Loading ML models...")
    try:
        flood_model, drought_model, heatwave_model = load_models()
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"⚠️ Model loading failed: {e} → using rule-based fallback for predictions")

initialize_models()

# City coordinates for map + predictions
CITY_COORDINATES = {
    "mumbai": {"lat": 19.0760, "lon": 72.8777},
    "delhi": {"lat": 28.7041, "lon": 77.1025},
    "bangalore": {"lat": 12.9716, "lon": 77.5946},
    "chennai": {"lat": 13.0827, "lon": 80.2707},
    "kolkata": {"lat": 22.5726, "lon": 88.3639},
    "hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "pune": {"lat": 18.5204, "lon": 73.8567},
    "ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "jaipur": {"lat": 26.9124, "lon": 75.7873},
    "lucknow": {"lat": 26.8467, "lon": 80.9462},
    "surat": {"lat": 21.1702, "lon": 72.8311},
    "kanpur": {"lat": 26.4499, "lon": 80.3319},
    "nagpur": {"lat": 21.1458, "lon": 79.0882},
    "indore": {"lat": 22.7196, "lon": 75.8577},
    "thane": {"lat": 19.2183, "lon": 72.9781},
    "bhopal": {"lat": 23.2599, "lon": 77.4126},
    "visakhapatnam": {"lat": 17.6868, "lon": 83.2185},
    "patna": {"lat": 25.5941, "lon": 85.1376},
    "vadodara": {"lat": 22.3072, "lon": 73.1812},
}

def get_coordinates(city_name):
    city_key = city_name.lower().strip()
    if city_key in CITY_COORDINATES:
        return CITY_COORDINATES[city_key]["lat"], CITY_COORDINATES[city_key]["lon"]
    return None, None

# ==================== ROUTES ====================

@app.route('/')
def index():
    # Changed to match your actual filename
    return send_from_directory('.', 'index_combined.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        city = data.get('city', '').strip()
        if not city:
            return jsonify({"success": False, "message": "City required"}), 400
        
        lat, lon = get_coordinates(city)
        if lat is None:
            lat = data.get('lat')
            lon = data.get('lon')
            if lat is None:
                return jsonify({"success": False, "message": f"Unknown city: {city}"}), 404
        
        # Using rule-based mode for more reliable output during development
        result = predict_disasters(city, float(lat), float(lon), use_ml=False)
        if result is None:
            return jsonify({"success": False, "message": "Prediction failed"}), 500
        
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/predict-all-cities', methods=['POST'])
def predict_all_cities():
    try:
        results = []
        for city_name, coords in CITY_COORDINATES.items():
            try:
                # Force rule-based for consistent map results
                prediction = predict_disasters(
                    city_name.title(),
                    coords["lat"],
                    coords["lon"],
                    use_ml=False
                )
                
                if prediction is None:
                    prediction = {
                        "predictions": {
                            "flood": {"probability_percent": 8, "risk_level": "Very Low"},
                            "drought": {"probability_percent": 12, "risk_level": "Low"},
                            "heatwave": {"probability_percent": 15, "risk_level": "Low"}
                        }
                    }

                probs = {k: v["probability_percent"] for k, v in prediction["predictions"].items()}
                max_risk = max(probs.values())
                main_type = max(probs, key=probs.get).title()

                results.append({
                    "city": city_name.title(),
                    "lat": coords["lat"],
                    "lon": coords["lon"],
                    "has_alert": max_risk >= 30,
                    "max_risk": max_risk,
                    "disaster_type": main_type,
                    "predictions": prediction["predictions"]
                })
                
            except Exception as e:
                print(f"City {city_name} failed: {e}")
                results.append({
                    "city": city_name.title(),
                    "lat": coords["lat"],
                    "lon": coords["lon"],
                    "has_alert": False,
                    "max_risk": 5.0,
                    "disaster_type": "Unknown",
                    "predictions": {}
                })
                continue
        
        return jsonify({
            "success": True,
            "cities": results,
            "total_analyzed": len(results)
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/sos', methods=['POST'])
def sos():
    data = request.json
    try:
        lat = float(data.get("latitude"))
        lon = float(data.get("longitude"))
        name = data.get("name", "Unknown User")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except:
        return jsonify({"error": "Invalid data"}), 400
    
    message = f"""🚨 EMERGENCY SOS ALERT 🚨

👤 Name: {name}
📍 Location: {lat}, {lon}
🕐 Time: {timestamp}

⚠️ Immediate assistance needed!"""
    
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message}
        )
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendLocation",
            json={"chat_id": TELEGRAM_CHAT_ID, "latitude": lat, "longitude": lon}
        )
        return jsonify({"status": "sent", "message": "SOS alert sent successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/cities', methods=['GET'])
def get_cities():
    cities = [
        {"name": c.title(), "key": c, "lat": coords["lat"], "lon": coords["lon"]}
        for c, coords in sorted(CITY_COORDINATES.items())
    ]
    return jsonify({"success": True, "cities": cities, "count": len(cities)})

if __name__ == '__main__':
    print("🌐 Server starting at: http://localhost:5000")
    print("Serving HTML file: index_combined.html")
    app.run(host='0.0.0.0', port=5000, debug=True)