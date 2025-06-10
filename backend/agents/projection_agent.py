"""
Projection Agent - Combines all factors into statistical projections for over/under betting
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
from data.sources.mlb_api import MLBDataSource
from models.ml.base_projections import BaseProjectionModel
from models.features.weather_adjustments import WeatherAdjustments
from models.ml.streak_detection import StreakDetector


class ProjectionAgent:
    """Agent responsible for generating player statistical projections"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            openai_api_key=settings.openai_api_key
        )
        self.mlb_data = MLBDataSource()
        self.base_model = BaseProjectionModel()
        self.weather_adj = WeatherAdjustments()
        self.streak_detector = StreakDetector()
        
        # Setup agent tools
        self.tools = [
            self._get_player_recent_stats,
            self._get_matchup_data,
            self._get_weather_forecast,
            self._calculate_base_projection,
            self._apply_adjustments
        ]
        
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create the LangChain agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a baseball projection specialist focused on over/under betting.
            
            Your goal is to generate accurate statistical projections for players that help determine 
            whether they will go OVER or UNDER specific lines on Underdog Fantasy.
            
            Key stat categories to focus on:
            - Hits, Total Bases, Runs, RBIs, Stolen Bases, Strikeouts (batting)
            - Strikeouts, Earned Runs, Hits Allowed (pitching)
            
            Process:
            1. Get player's recent performance (last 15 games)
            2. Analyze pitcher-batter matchup if applicable
            3. Get weather conditions for today's game
            4. Calculate base projection using ML model
            5. Apply all adjustment factors
            6. Return final projection with confidence level
            
            Always provide reasoning for your projections and confidence levels.
            """),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    @tool
    def _get_player_recent_stats(self, player_id: int, days: int = 15) -> Dict:
        """Get player's recent performance statistics"""
        try:
            stats = self.mlb_data.get_player_recent_stats(player_id, days)
            return {
                "player_id": player_id,
                "recent_stats": stats,
                "games_played": len(stats),
                "avg_performance": {
                    "hits": np.mean([g["hits"] for g in stats]),
                    "total_bases": np.mean([g["total_bases"] for g in stats]),
                    "runs": np.mean([g["runs"] for g in stats]),
                    "rbis": np.mean([g["rbis"] for g in stats]),
                    "strikeouts": np.mean([g["strikeouts"] for g in stats])
                }
            }
        except Exception as e:
            return {"error": f"Failed to get recent stats: {str(e)}"}
    
    @tool
    def _get_matchup_data(self, batter_id: int, pitcher_id: int) -> Dict:
        """Get historical matchup data between batter and pitcher"""
        try:
            matchup = self.mlb_data.get_batter_pitcher_matchup(batter_id, pitcher_id)
            return {
                "historical_at_bats": matchup.get("at_bats", 0),
                "historical_performance": matchup.get("stats", {}),
                "pitcher_handedness": matchup.get("pitcher_hand"),
                "batter_vs_handedness": matchup.get("batter_vs_hand_stats", {}),
                "recent_form": matchup.get("last_10_games", {})
            }
        except Exception as e:
            return {"error": f"Failed to get matchup data: {str(e)}"}
    
    @tool
    def _get_weather_forecast(self, game_id: str) -> Dict:
        """Get weather forecast for the game"""
        try:
            weather = self.mlb_data.get_game_weather(game_id)
            return {
                "temperature": weather.get("temperature"),
                "wind_speed": weather.get("wind_speed"),
                "wind_direction": weather.get("wind_direction"),
                "humidity": weather.get("humidity"),
                "pressure": weather.get("pressure"),
                "conditions": weather.get("conditions")
            }
        except Exception as e:
            return {"error": f"Failed to get weather data: {str(e)}"}
    
    @tool
    def _calculate_base_projection(self, player_data: Dict) -> Dict:
        """Calculate base projection using ML model"""
        try:
            projection = self.base_model.predict(player_data)
            return {
                "hits": projection["hits"],
                "total_bases": projection["total_bases"],
                "runs": projection["runs"],
                "rbis": projection["rbis"],
                "strikeouts": projection["strikeouts"],
                "confidence": projection["confidence"]
            }
        except Exception as e:
            return {"error": f"Failed to calculate base projection: {str(e)}"}
    
    @tool
    def _apply_adjustments(self, base_projection: Dict, weather: Dict, matchup: Dict) -> Dict:
        """Apply weather and matchup adjustments to base projection"""
        try:
            adjusted = base_projection.copy()
            
            # Weather adjustments
            if weather.get("temperature"):
                temp_factor = self.weather_adj.temperature_effect(
                    weather["temperature"], "total_bases"
                )
                adjusted["total_bases"] *= temp_factor
                adjusted["hits"] *= (temp_factor * 0.5 + 0.5)  # Smaller effect on hits
            
            # Wind adjustments
            if weather.get("wind_speed") and weather.get("wind_direction"):
                wind_factor = self.weather_adj.wind_effect(
                    weather["wind_speed"], weather["wind_direction"]
                )
                adjusted["total_bases"] *= wind_factor
            
            # Matchup adjustments
            if matchup.get("batter_vs_handedness"):
                platoon_factor = matchup["batter_vs_handedness"].get("ops_factor", 1.0)
                for stat in ["hits", "total_bases", "runs", "rbis"]:
                    adjusted[stat] *= platoon_factor
            
            return adjusted
        except Exception as e:
            return {"error": f"Failed to apply adjustments: {str(e)}"}
    
    def generate_projection(self, player_id: int, game_id: str, stat_type: str) -> Dict:
        """Generate complete projection for a player and stat"""
        query = f"""
        Generate a projection for player {player_id} in game {game_id} for the {stat_type} stat.
        
        Please follow this process:
        1. Get the player's recent stats (last 15 games)
        2. Get matchup data if this is a hitting stat
        3. Get weather forecast for the game
        4. Calculate base projection
        5. Apply all relevant adjustments
        6. Provide final projection with confidence level (1-5)
        
        Focus on providing actionable insights for over/under betting decisions.
        """
        
        result = self.agent.invoke({"input": query})
        return result["output"]


# Standalone function for quick projections
def quick_projection(player_id: int, game_id: str, stat_type: str) -> Dict:
    """Quick projection without agent overhead"""
    agent = ProjectionAgent()
    return agent.generate_projection(player_id, game_id, stat_type)
