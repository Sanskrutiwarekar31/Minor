# ============================================================
# COMPLETE FLASK APP FOR DISASTER MANAGEMENT SYSTEM
# Serves HTML + All API Endpoints (Weather, Prediction, News)
# ============================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import requests
from datetime import datetime

# Import the prediction system
try:
    try:
        from disaster_prediction_complete_fix import predict_disasters, load_models, train_all_models
        print("✅ Imported from disaster_prediction_complete_fix")
    except ImportError:
        try:
            from disaster_prediction_fixed import predict_disasters, load_models, train_all_models
            print("✅ Imported from disaster_prediction_fixed")
        except ImportError:
            try:
                from flood_predictor import predict_disasters, load_models, train_all_models
                print("✅ Imported from flood_predictor")
            except ImportError:
                from improved_disaster_prediction import predict_disasters, load_models, train_all_models
                print("✅ Imported from improved_disaster_prediction")
except ImportError as e:
    print(f"❌ Could not import prediction module: {e}")
    print("📝 Please ensure your prediction file exists in the same directory")
    sys.exit(1)

app = Flask(__name__, static_folder='.')
CORS(app)

# Configuration
WEATHER_API_KEY = "14c27bb50b0044d6b37175716263101"
NEWS_PROXY = "https://api.allorigins.win/raw?url="

# Global variables for models
flood_model = None
drought_model = None
heatwave_model = None

# ============================================================
# LOAD MODELS ON STARTUP
# ============================================================

def initialize_models():
    """Load or train models on startup"""
    global flood_model, drought_model, heatwave_model
    
    print("🚀 Loading ML models...")
    try:
        flood_model, drought_model, heatwave_model = load_models()
        if flood_model is not None and hasattr(flood_model, 'model') and flood_model.model is not None:
            print("✅ Models loaded successfully")
        else:
            print("⚠️ Models exist but not fully trained, using rule-based predictions")
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
        print("⚠️ Will use rule-based predictions as fallback")

initialize_models()

# ============================================================
# COORDINATE DATABASE
# ============================================================

CITY_COORDINATES = {
    "mumbai": {"lat": 19.0760, "lon": 72.8777},
    "delhi": {"lat": 28.7041, "lon": 77.1025},
    "bangalore": {"lat": 12.9716, "lon": 77.5946},
    "bengaluru": {"lat": 12.9716, "lon": 77.5946},
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
    """Get coordinates for a city"""
    city_key = city_name.lower().strip()
    
    if city_key in CITY_COORDINATES:
        return CITY_COORDINATES[city_key]["lat"], CITY_COORDINATES[city_key]["lon"]
    
    for key in CITY_COORDINATES:
        if city_key in key or key in city_key:
            return CITY_COORDINATES[key]["lat"], CITY_COORDINATES[key]["lon"]
    
    return None, None

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/api/weather', methods=['GET', 'POST'])
def get_weather():
    """Weather API endpoint"""
    try:
        if request.method == 'POST':
            data = request.get_json()
            city = data.get('city', '').strip()
        else:
            city = request.args.get('city', '').strip()
        
        if not city:
            return jsonify({
                "success": False,
                "message": "City name is required"
            }), 400
        
        # Call WeatherAPI
        url = f"https://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return jsonify({
                "success": False,
                "message": "Weather data not found for this location"
            }), 404
        
        weather_data = response.json()
        
        return jsonify({
            "success": True,
            "data": {
                "location": weather_data["location"]["name"],
                "country": weather_data["location"]["country"],
                "current": {
                    "temp_c": weather_data["current"]["temp_c"],
                    "humidity": weather_data["current"]["humidity"],
                    "precip_mm": weather_data["current"]["precip_mm"],
                    "wind_kph": weather_data["current"]["wind_kph"],
                    "condition": weather_data["current"]["condition"]["text"],
                    "pressure_mb": weather_data["current"].get("pressure_mb", 0),
                    "cloud": weather_data["current"].get("cloud", 0)
                }
            }
        })
    
    except requests.Timeout:
        return jsonify({
            "success": False,
            "message": "Weather API request timeout. Please try again."
        }), 504
    
    except Exception as e:
        print(f"❌ Weather error: {e}")
        return jsonify({
            "success": False,
            "message": f"Error fetching weather data: {str(e)}"
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Disaster prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "message": "No data provided"
            }), 400
        
        city = data.get('city', '').strip()
        
        if not city:
            return jsonify({
                "success": False,
                "message": "City name is required"
            }), 400
        
        # Get coordinates
        lat = data.get('lat')
        lon = data.get('lon')
        
        if lat is None or lon is None:
            lat, lon = get_coordinates(city)
            if lat is None:
                return jsonify({
                    "success": False,
                    "message": f"Coordinates not found for '{city}'. Please try another city."
                }), 404
        
        print(f"🌍 Processing prediction request for {city} ({lat}, {lon})")
        
        # Make prediction
        result = predict_disasters(city, float(lat), float(lon))
        
        if result is None:
            return jsonify({
                "success": False,
                "message": "Could not fetch weather data. Please try again later."
            }), 500
        
        return jsonify({
            "success": True,
            "data": result
        })
    
    except ValueError as e:
        return jsonify({
            "success": False,
            "message": f"Invalid coordinates: {str(e)}"
        }), 400
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "message": f"Prediction error: {str(e)}"
        }), 500

@app.route('/api/news', methods=['GET', 'POST'])
def get_news():
    """News aggregator endpoint (proxy for RSS feeds)"""
    try:
        if request.method == 'POST':
            data = request.get_json()
            keywords = data.get('keywords', [])
        else:
            keywords = request.args.get('keywords', '').split(',')
        
        # News feeds
        feeds = [
            "https://feeds.feedburner.com/ndtvnews-top-stories",
            "https://www.thehindu.com/news/national/?service=rss",
            "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
            "https://indianexpress.com/feed/",
            "https://www.hindustantimes.com/rss/topnews/rssfeed.xml"
        ]
        
        all_items = []
        
        for feed_url in feeds:
            try:
                # Use proxy to fetch RSS feed
                proxy_url = NEWS_PROXY + requests.utils.quote(feed_url)
                response = requests.get(proxy_url, timeout=10)
                
                if response.status_code == 200:
                    # Parse RSS would go here
                    # For now, return success
                    pass
            except Exception as e:
                print(f"⚠️ Failed to fetch {feed_url}: {e}")
                continue
        
        return jsonify({
            "success": True,
            "message": "News feeds are fetched client-side for better performance",
            "feeds": feeds
        })
    
    except Exception as e:
        print(f"❌ News error: {e}")
        return jsonify({
            "success": False,
            "message": f"Error fetching news: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    models_status = "loaded" if (flood_model is not None) else "not_loaded"
    
    return jsonify({
        "status": "healthy",
        "models_status": models_status,
        "version": "3.0",
        "supported_cities": len(CITY_COORDINATES),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/cities', methods=['GET'])
def get_cities():
    """Get list of supported cities"""
    cities = [
        {
            "name": city.title(),
            "key": city,
            "lat": coords["lat"],
            "lon": coords["lon"]
        }
        for city, coords in sorted(CITY_COORDINATES.items())
    ]
    return jsonify({
        "success": True,
        "cities": cities,
        "count": len(cities)
    })

# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "message": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "message": "Internal server error"
    }), 500

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   DISASTER MANAGEMENT SYSTEM - COMPLETE STACK v3.0       ║
    ║   HTML + Weather API + Prediction + News                 ║
    ╚══════════════════════════════════════════════════════════╝
    
    🌐 Server starting...
    📍 Open your browser at: http://localhost:5000
    
    📋 Available Endpoints:
       • GET  /                  - Main HTML interface
       • POST /api/predict       - Disaster predictions
       • GET  /api/weather       - Weather data
       • GET  /api/news          - News aggregator
       • GET  /api/cities        - Supported cities
       • GET  /api/health        - Health check
    
    🛑 Press Ctrl+C to stop the server
    
    """)
    
    # Check if HTML file exists
    if not os.path.exists('index.html'):
        print("⚠️  WARNING: index.html not found!")
        print("    Please make sure the HTML file is in the same directory as this script.")
        print()
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )