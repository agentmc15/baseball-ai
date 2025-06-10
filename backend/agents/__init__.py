"""
Baseball AI Prediction Agents
"""
from .projection_agent import ProjectionAgent, quick_projection
from .line_value_agent import LineValueAgent, analyze_value
from .weather_agent import WeatherAgent, quick_weather_analysis
from .correlation_agent import CorrelationAgent, quick_correlation_analysis

__all__ = [
    "ProjectionAgent",
    "LineValueAgent", 
    "WeatherAgent",
    "CorrelationAgent",
    "quick_projection",
    "analyze_value",
    "quick_weather_analysis",
    "quick_correlation_analysis"
]
