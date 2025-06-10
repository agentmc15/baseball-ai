"""
Weather Impact Agent - Analyzes environmental factors affecting player performance
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, date
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from utils.config import settings
from data.sources.weather_api import WeatherAPI
from models.features.weather_adjustments import WeatherAdjustments


class WeatherAgent:
    """Agent responsible for weather impact analysis"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            openai_api_key=settings.openai_api_key
        )
        self.weather_api = WeatherAPI()
        self.adjustments = WeatherAdjustments()
        
        self.tools = [
            self._get_current_weather,
            self._get_weather_forecast,
            self._calculate_temperature_effect,
            self._calculate_wind_effect,
            self._calculate_humidity_effect,
            self._get_historical_weather_performance,
            self._calculate_ballpark_factors
        ]
        
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create the LangChain agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a weather impact specialist for baseball analytics.
            
            Your expertise is in analyzing how weather conditions affect player performance,
            specifically for over/under betting decisions.
            
            Key factors to analyze:
            1. Temperature effects on ball flight (+2.5 feet per 10°F)
            2. Wind speed and direction impact
            3. Humidity effects on ball movement
            4. Atmospheric pressure effects
            5. Day vs night games
            6. Historical performance in similar conditions
            
            Focus on:
            - Hits, Total Bases (most affected by weather)
            - Home runs (significantly affected)
            - Runs/RBIs (indirectly affected)
            - Strikeouts (pitcher effectiveness changes)
            
            Provide clear adjustment factors and confidence levels.
            """),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    @tool
    def _get_current_weather(self, venue_id: int) -> Dict:
        """Get current weather conditions at the ballpark"""
        try:
            weather = self.weather_api.get_current_weather(venue_id)
            return {
                "temperature": weather.get("temperature"),
                "feels_like": weather.get("feels_like"),
                "humidity": weather.get("humidity"),
                "pressure": weather.get("pressure"),
                "wind_speed": weather.get("wind_speed"),
                "wind_direction": weather.get("wind_direction"),
                "conditions": weather.get("conditions"),
                "visibility": weather.get("visibility")
            }
        except Exception as e:
            return {"error": f"Failed to get current weather: {str(e)}"}
    
    @tool
    def _get_weather_forecast(self, venue_id: int, game_time: str) -> Dict:
        """Get weather forecast for game time"""
        try:
            forecast = self.weather_api.get_game_forecast(venue_id, game_time)
            return {
                "game_temp": forecast.get("temperature"),
                "wind_speed": forecast.get("wind_speed"),
                "wind_direction": forecast.get("wind_direction"),
                "humidity": forecast.get("humidity"),
                "pressure": forecast.get("pressure"),
                "precipitation_chance": forecast.get("precipitation_chance"),
                "conditions": forecast.get("conditions"),
                "confidence": forecast.get("forecast_confidence")
            }
        except Exception as e:
            return {"error": f"Failed to get weather forecast: {str(e)}"}
    
    @tool
    def _calculate_temperature_effect(self, temperature: float, stat_type: str, baseline_temp: float = 70) -> Dict:
        """Calculate temperature effect on offensive stats"""
        try:
            effect = self.adjustments.temperature_effect(temperature, stat_type)
            
            # More detailed analysis
            temp_diff = temperature - baseline_temp
            
            # Temperature effects (based on physics and historical data)
            effects = {
                "total_bases": 1 + (temp_diff / 10) * 0.025,  # +2.5% per 10°F
                "hits": 1 + (temp_diff / 10) * 0.015,        # Smaller effect
                "home_runs": 1 + (temp_diff / 10) * 0.04,    # Larger effect
                "runs": 1 + (temp_diff / 10) * 0.02,
                "rbis": 1 + (temp_diff / 10) * 0.02
            }
            
            return {
                "temperature": temperature,
                "baseline_temp": baseline_temp,
                "temp_difference": temp_diff,
                "effect_factor": effects.get(stat_type, 1.0),
                "stat_type": stat_type,
                "reasoning": f"Every 10°F above {baseline_temp}°F increases offensive stats due to reduced air density"
            }
        except Exception as e:
            return {"error": f"Failed to calculate temperature effect: {str(e)}"}
    
    @tool
    def _calculate_wind_effect(self, wind_speed: float, wind_direction: str, ballpark: str) -> Dict:
        """Calculate wind effect on offensive stats"""
        try:
            effect = self.adjustments.wind_effect(wind_speed, wind_direction, ballpark)
            
            # Wind direction mapping
            direction_effects = {
                "out_to_left": {"total_bases": 1.1, "home_runs": 1.2},
                "out_to_right": {"total_bases": 1.1, "home_runs": 1.2},
                "out_to_center": {"total_bases": 1.15, "home_runs": 1.3},
                "in_from_left": {"total_bases": 0.95, "home_runs": 0.85},
                "in_from_right": {"total_bases": 0.95, "home_runs": 0.85},
                "in_from_center": {"total_bases": 0.9, "home_runs": 0.8},
                "cross_wind": {"total_bases": 1.0, "home_runs": 0.95}
            }
            
            # Wind speed multiplier
            speed_multiplier = min(1 + (wind_speed - 5) * 0.02, 1.4)  # Cap at 40% increase
            
            base_effects = direction_effects.get(wind_direction, {"total_bases": 1.0, "home_runs": 1.0})
            final_effects = {k: v * speed_multiplier for k, v in base_effects.items()}
            
            return {
                "wind_speed": wind_speed,
                "wind_direction": wind_direction,
                "ballpark": ballpark,
                "effect_factors": final_effects,
                "speed_multiplier": speed_multiplier,
                "reasoning": f"{wind_speed} mph wind {wind_direction} at {ballpark}"
            }
        except Exception as e:
            return {"error": f"Failed to calculate wind effect: {str(e)}"}
    
    @tool
    def _calculate_humidity_effect(self, humidity: float, temperature: float) -> Dict:
        """Calculate humidity effect on ball flight"""
        try:
            # High humidity = thicker air = less ball flight
            # But also affects pitcher grip and ball movement
            
            baseline_humidity = 50
            humidity_diff = humidity - baseline_humidity
            
            # Humidity effects
            ball_flight_factor = 1 - (humidity_diff / 100) * 0.1  # Max 10% reduction at 100% humidity
            pitcher_factor = 1 + (humidity_diff / 100) * 0.05    # Worse grip in high humidity
            
            return {
                "humidity": humidity,
                "temperature": temperature,
                "ball_flight_factor": ball_flight_factor,
                "pitcher_effectiveness": pitcher_factor,
                "combined_effect": ball_flight_factor / pitcher_factor,
                "reasoning": f"{humidity}% humidity affects both ball flight and pitcher grip"
            }
        except Exception as e:
            return {"error": f"Failed to calculate humidity effect: {str(e)}"}
    
    @tool
    def _get_historical_weather_performance(self, player_id: int, weather_conditions: Dict) -> Dict:
        """Get player's historical performance in similar weather"""
        try:
            # This would query historical database for similar conditions
            # For now, return simulated data structure
            
            temp_range = (weather_conditions["temperature"] - 10, weather_conditions["temperature"] + 10)
            wind_range = (weather_conditions["wind_speed"] - 5, weather_conditions["wind_speed"] + 5)
            
            historical_data = {
                "games_in_conditions": 12,
                "avg_performance": {
                    "hits": 1.2,
                    "total_bases": 1.8,
                    "runs": 0.8,
                    "rbis": 0.9
                },
                "performance_vs_season": {
                    "hits": 1.05,     # 5% better in these conditions
                    "total_bases": 1.1,
                    "runs": 0.95,
                    "rbis": 1.0
                },
                "temperature_range": temp_range,
                "wind_range": wind_range,
                "sample_size_confidence": 0.8
            }
            
            return historical_data
        except Exception as e:
            return {"error": f"Failed to get historical weather performance: {str(e)}"}
    
    @tool
    def _calculate_ballpark_factors(self, venue_id: int, weather: Dict) -> Dict:
        """Calculate ballpark-specific weather effects"""
        try:
            # Ballpark dimensions and characteristics
            ballpark_data = {
                1: {"name": "Yankee Stadium", "elevation": 55, "fair_territory": "small"},
                2: {"name": "Fenway Park", "elevation": 21, "fair_territory": "small"},
                3: {"name": "Coors Field", "elevation": 5200, "fair_territory": "large"},
                # Add more ballparks...
            }
            
            park = ballpark_data.get(venue_id, {"elevation": 500, "fair_territory": "medium"})
            
            # Elevation effect (higher = more offense)
            elevation_factor = 1 + (park["elevation"] / 1000) * 0.02
            
            # Fair territory effect with wind
            territory_wind_interaction = {
                "small": 1.1,   # Wind has more effect in smaller parks
                "medium": 1.0,
                "large": 0.9    # Wind dispersed in larger parks
            }
            
            wind_park_factor = territory_wind_interaction.get(park["fair_territory"], 1.0)
            
            return {
                "venue_id": venue_id,
                "ballpark_name": park["name"],
                "elevation": park["elevation"],
                "elevation_factor": elevation_factor,
                "wind_park_interaction": wind_park_factor,
                "combined_park_factor": elevation_factor * wind_park_factor,
                "reasoning": f"{park['name']} at {park['elevation']} feet elevation with {park['fair_territory']} fair territory"
            }
        except Exception as e:
            return {"error": f"Failed to calculate ballpark factors: {str(e)}"}
    
    def analyze_weather_impact(self, venue_id: int, game_time: str, player_id: int, stat_type: str) -> Dict:
        """Complete weather impact analysis"""
        
        query = f"""
        Analyze weather impact for:
        - Venue: {venue_id}
        - Game time: {game_time}
        - Player: {player_id}
        - Stat: {stat_type}
        
        Please:
        1. Get current weather and forecast for game time
        2. Calculate temperature effects on {stat_type}
        3. Calculate wind effects (speed and direction)
        4. Calculate humidity and pressure effects
        5. Get historical player performance in similar conditions
        6. Calculate ballpark-specific factors
        7. Provide final weather adjustment factor and confidence
        
        Focus on how these conditions will affect {stat_type} specifically.
        """
        
        result = self.agent.invoke({"input": query})
        return result["output"]


def quick_weather_analysis(venue_id: int, game_time: str, stat_type: str) -> Dict:
    """Quick weather analysis without player-specific data"""
    agent = WeatherAgent()
    return agent.analyze_weather_impact(venue_id, game_time, 0, stat_type)
