import streamlit as st
import folium
from streamlit_folium import st_folium
import requests

st.set_page_config(page_title="Disaster Alert System", layout="centered")
st.title("🌍 Disaster Alert System (Location-Based with Alerts)")

API_KEY = "960ed8bd981cd018c47ed8c49b2152bb"

def get_user_location():
    try:
        res = requests.get("https://ipinfo.io/json", timeout=5)
        data = res.json()
        city = data.get("city")
        loc = data.get("loc")
        if city and loc:
            lat, lon = map(float, loc.split(","))
            return city, lat, lon
    except:
        pass
    return None, None, None

def get_alert_level(rain_mm):
    if rain_mm > 150:
        return "RED", "🔴 EMERGENCY ALERT: Heavy rainfall detected. High flood risk! Stay indoors and follow authorities."
    elif rain_mm > 50:
        return "ORANGE", "🟠 WARNING ALERT: Moderate rainfall detected. Stay cautious."
    else:
        return "GREEN", "🟢 SAFE STATUS: Weather conditions are normal."

city, lat, lon = get_user_location()

st.write("📍 Auto-detected location:", city)

manual_location = st.text_input("Or enter your city manually:")

if manual_location:
    location = manual_location
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
elif city:
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
else:
    weather_url = None

if weather_url:
    response = requests.get(weather_url)

    if response.status_code == 200:
        data = response.json()
        lat = data["coord"]["lat"]
        lon = data["coord"]["lon"]
        rain_mm = data.get("rain", {}).get("1h", 0)

        alert_level, alert_message = get_alert_level(rain_mm)
        color = {"RED": "red", "ORANGE": "orange", "GREEN": "green"}[alert_level]

        m = folium.Map(location=[lat, lon], zoom_start=10)
        folium.Marker(
            [lat, lon],
            popup=f"{location if manual_location else city}\nRainfall: {rain_mm} mm\n{alert_message}",
            icon=folium.Icon(color=color, icon="warning-sign"),
        ).add_to(m)

        st_folium(m, width=700, height=500)

        if alert_level == "RED":
            st.error(alert_message)
        elif alert_level == "ORANGE":
            st.warning(alert_message)
        else:
            st.success(alert_message)

        if st.button("📩 Send Alert Message"):
            st.success("✅ Alert message sent successfully!")
            st.info(f"📨 Message content:\n\n{alert_message}")

    else:
        st.error("❌ Weather API error. Check API key or city name.")
else:
    st.info("⏳ Detecting your location... or enter it manually.")
