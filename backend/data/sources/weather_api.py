"""
Weather API Data Source using OpenWeather API
"""
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional

from utils.config import settings


class WeatherAPI:
    """Interface to OpenWeather API for ballpark weather"""
    
    def __init__(self):
        self.api_key = settings.openweather_api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        # MLB ballpark coordinates (sample)
        self.ballpark_coords = {
            1: {"lat": 40.8296, "lon": -73.9262, "name": "Yankee Stadium"},
            2: {"lat": 42.3467, "lon": -71.0972, "name": "Fenway Park"},
            3: {"lat": 39.7560, "lon": -104.9942, "name": "Coors Field"},
            # Add more ballparks as needed
        }
    
    def get_current_weather(self, venue_id: int) -> Dict:
        """Get current weather at ballpark"""
        coords = self.ballpark_coords.get(venue_id)
        if not coords:
            return {"error": "Unknown venue"}
        
        params = {
            "lat": coords["lat"],
            "lon": coords["lon"],
            "appid": self.api_key,
            "units": "imperial"
        }
        
        try:
            response = requests.get(f"{self.base_url}/weather", params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_speed": data.get("wind", {}).get("speed", 0),
                "wind_direction": self._convert_wind_direction(data.get("wind", {}).get("deg", 0)),
                "conditions": data["weather"][0]["description"],
                "visibility": data.get("visibility", 10000) / 1000  # Convert to km
            }
        except requests.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return {"error": str(e)}
    
    def get_game_forecast(self, venue_id: int, game_time: str) -> Dict:
        """Get weather forecast for game time"""
        coords = self.ballpark_coords.get(venue_id)
        if not coords:
            return {"error": "Unknown venue"}
        
        params = {
            "lat": coords["lat"],
            "lon": coords["lon"],
            "appid": self.api_key,
            "units": "imperial"
        }
        
        try:
            response = requests.get(f"{self.base_url}/forecast", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Find forecast closest to game time
            game_dt = datetime.fromisoformat(game_time.replace('Z', '+00:00'))
            
            closest_forecast = None
            min_time_diff = float('inf')
            
            for forecast in data.get("list", []):
                forecast_dt = datetime.fromtimestamp(forecast["dt"])
                time_diff = abs((game_dt - forecast_dt).total_seconds())
                
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_forecast = forecast
            
            if closest_forecast:
                return {
                    "temperature": closest_forecast["main"]["temp"],
                    "humidity": closest_forecast["main"]["humidity"],
                    "pressure": closest_forecast["main"]["pressure"],
                    "wind_speed": closest_forecast.get("wind", {}).get("speed", 0),
                    "wind_direction": self._convert_wind_direction(closest_forecast.get("wind", {}).get("deg", 0)),
                    "conditions": closest_forecast["weather"][0]["description"],
                    "precipitation_chance": closest_forecast.get("pop", 0) * 100,
                    "forecast_confidence": 0.8 if min_time_diff < 10800 else 0.6  # 3 hours
                }
            else:
                return {"error": "No suitable forecast found"}
                
        except requests.RequestException as e:
            print(f"Error fetching forecast data: {e}")
            return {"error": str(e)}
    
    def _convert_wind_direction(self, degrees: float) -> str:
        """Convert wind degrees to ballpark-relative direction"""
        # Simplified conversion - would need ballpark-specific orientation
        if 315 <= degrees or degrees < 45:
            return "out_to_center"
        elif 45 <= degrees < 135:
            return "out_to_right"
        elif 135 <= degrees < 225:
            return "in_from_center"
        else:
            return "out_to_left"
