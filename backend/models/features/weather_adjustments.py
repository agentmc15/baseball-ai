"""
Weather adjustment calculations for player projections
"""
import math
from typing import Dict


class WeatherAdjustments:
    """Calculate weather-based adjustments to player projections"""
    
    def temperature_effect(self, temp_f: float, stat_type: str, baseline_temp: float = 70) -> float:
        """
        Calculate temperature effect on offensive stats
        Based on physics: warmer air is less dense, ball travels farther
        """
        temp_diff = temp_f - baseline_temp
        
        # Temperature effects by stat type
        effects = {
            "hits": 1 + (temp_diff / 10) * 0.015,        # 1.5% per 10°F
            "total_bases": 1 + (temp_diff / 10) * 0.025, # 2.5% per 10°F  
            "home_runs": 1 + (temp_diff / 10) * 0.04,    # 4% per 10°F
            "runs": 1 + (temp_diff / 10) * 0.02,         # 2% per 10°F
            "rbis": 1 + (temp_diff / 10) * 0.02,         # 2% per 10°F
            "strikeouts": 1 - (temp_diff / 10) * 0.01    # Slight decrease (better hitting)
        }
        
        return effects.get(stat_type, 1.0)
    
    def wind_effect(self, wind_speed: float, wind_direction: str, ballpark: str = "generic") -> float:
        """
        Calculate wind effect on offensive stats
        """
        # Base wind effects by direction
        direction_multipliers = {
            "out_to_left": 1.1,      # Helping wind to left field
            "out_to_right": 1.1,     # Helping wind to right field  
            "out_to_center": 1.15,   # Helping wind to center (best for offense)
            "in_from_left": 0.95,    # Hurting wind from left
            "in_from_right": 0.95,   # Hurting wind from right
            "in_from_center": 0.9,   # Hurting wind from center (worst for offense)
            "cross_wind": 1.0        # Neutral crosswind
        }
        
        base_effect = direction_multipliers.get(wind_direction, 1.0)
        
        # Wind speed effect (diminishing returns)
        # Light wind (0-5 mph): minimal effect
        # Moderate wind (6-15 mph): significant effect
        # Strong wind (16+ mph): major effect but capped
        
        if wind_speed <= 5:
            speed_multiplier = 1.0
        elif wind_speed <= 15:
            speed_multiplier = 1 + (wind_speed - 5) * 0.02  # 2% per mph
        else:
            speed_multiplier = 1.2 + (wind_speed - 15) * 0.01  # Diminishing returns
        
        # Cap maximum effect
        final_effect = base_effect * min(speed_multiplier, 1.4)
        
        return final_effect
    
    def humidity_effect(self, humidity: float, temperature: float) -> Dict[str, float]:
        """
        Calculate humidity effects on performance
        High humidity = thicker air = less ball flight, but also affects pitcher grip
        """
        # Ball flight effect (higher humidity = less distance)
        ball_flight_factor = 1 - (humidity - 50) / 100 * 0.1  # Max 10% reduction at 100% humidity
        
        # Pitcher effectiveness (higher humidity = worse grip = less control)
        pitcher_factor = 1 + (humidity - 50) / 100 * 0.05  # Max 5% worse control
        
        # Combined effect on hitting stats
        hitting_effect = ball_flight_factor / pitcher_factor
        
        return {
            "hitting_stats": hitting_effect,
            "ball_flight": ball_flight_factor,
            "pitcher_effectiveness": 1 / pitcher_factor
        }
    
    def pressure_effect(self, pressure_inHg: float) -> float:
        """
        Calculate atmospheric pressure effect
        Lower pressure = thinner air = more offense
        """
        standard_pressure = 29.92  # Standard atmospheric pressure
        pressure_diff = standard_pressure - pressure_inHg
        
        # 1% change in offense per 0.1 inHg difference
        effect = 1 + pressure_diff * 0.1
        
        return max(0.8, min(1.2, effect))  # Cap between 80% and 120%
    
    def ballpark_wind_interaction(self, wind_speed: float, wind_direction: str, 
                                 ballpark_factors: Dict) -> float:
        """
        Calculate ballpark-specific wind interactions
        """
        # Get ballpark characteristics
        dimensions = ballpark_factors.get("dimensions", {})
        elevation = ballpark_factors.get("elevation", 0)
        
        # Elevation effect on wind (higher altitude = more wind effect)
        elevation_multiplier = 1 + (elevation / 1000) * 0.05  # 5% per 1000 feet
        
        # Ballpark size effect on wind
        size_factor = dimensions.get("fair_territory", "medium")
        size_multipliers = {
            "small": 1.1,    # Wind has more effect in smaller parks
            "medium": 1.0,
            "large": 0.9     # Wind effect dispersed in larger parks
        }
        
        size_multiplier = size_multipliers.get(size_factor, 1.0)
        
        # Wall height effect (higher walls reduce wind help)
        wall_height = dimensions.get("avg_wall_height", 8)  # feet
        wall_factor = 1 - max(0, (wall_height - 8) / 20)  # Reduce effect for tall walls
        
        combined_effect = elevation_multiplier * size_multiplier * wall_factor
        
        return combined_effect
    
    def day_night_adjustment(self, is_day_game: bool, temperature: float) -> float:
        """
        Calculate day vs night game adjustments
        """
        if is_day_game:
            # Day games: better visibility but potentially hotter
            visibility_bonus = 1.02  # 2% bonus for better visibility
            
            if temperature > 85:
                heat_penalty = 0.98  # 2% penalty for extreme heat
            else:
                heat_penalty = 1.0
                
            return visibility_bonus * heat_penalty
        else:
            # Night games: cooler but potentially worse visibility/dew
            return 0.98  # Slight penalty for night conditions
    
    def calculate_total_weather_adjustment(self, weather_data: Dict, stat_type: str, 
                                         ballpark_data: Dict = None) -> Dict:
        """
        Calculate combined weather adjustment for a stat
        """
        adjustments = {
            "temperature": 1.0,
            "wind": 1.0,
            "humidity": 1.0,
            "pressure": 1.0,
            "day_night": 1.0,
            "ballpark_interaction": 1.0
        }
        
        # Temperature adjustment
        if "temperature" in weather_data:
            adjustments["temperature"] = self.temperature_effect(
                weather_data["temperature"], stat_type
            )
        
        # Wind adjustment
        if "wind_speed" in weather_data and "wind_direction" in weather_data:
            adjustments["wind"] = self.wind_effect(
                weather_data["wind_speed"], 
                weather_data["wind_direction"]
            )
        
        # Humidity adjustment
        if "humidity" in weather_data and "temperature" in weather_data:
            humidity_effects = self.humidity_effect(
                weather_data["humidity"], 
                weather_data["temperature"]
            )
            adjustments["humidity"] = humidity_effects["hitting_stats"]
        
        # Pressure adjustment
        if "pressure" in weather_data:
            adjustments["pressure"] = self.pressure_effect(weather_data["pressure"])
        
        # Day/night adjustment
        if "is_day_game" in weather_data and "temperature" in weather_data:
            adjustments["day_night"] = self.day_night_adjustment(
                weather_data["is_day_game"],
                weather_data["temperature"]
            )
        
        # Ballpark interaction
        if ballpark_data and "wind_speed" in weather_data:
            adjustments["ballpark_interaction"] = self.ballpark_wind_interaction(
                weather_data["wind_speed"],
                weather_data.get("wind_direction", "cross_wind"),
                ballpark_data
            )
        
        # Calculate total adjustment
        total_adjustment = 1.0
        for factor, value in adjustments.items():
            total_adjustment *= value
        
        # Cap total adjustment to reasonable range
        total_adjustment = max(0.7, min(1.4, total_adjustment))
        
        return {
            "total_adjustment": total_adjustment,
            "individual_adjustments": adjustments,
            "stat_type": stat_type,
            "confidence": self._calculate_weather_confidence(weather_data)
        }
    
    def _calculate_weather_confidence(self, weather_data: Dict) -> float:
        """Calculate confidence in weather adjustments"""
        confidence_factors = []
        
        # Temperature confidence (higher confidence for extreme temps)
        if "temperature" in weather_data:
            temp = weather_data["temperature"]
            if temp < 50 or temp > 90:
                confidence_factors.append(0.9)  # High confidence for extreme temps
            else:
                confidence_factors.append(0.7)  # Moderate confidence for normal temps
        
        # Wind confidence (higher confidence for strong winds)
        if "wind_speed" in weather_data:
            wind = weather_data["wind_speed"]
            if wind > 15:
                confidence_factors.append(0.85)
            elif wind > 8:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
        
        # Default confidence if no factors
        if not confidence_factors:
            return 0.5
        
        return sum(confidence_factors) / len(confidence_factors)
