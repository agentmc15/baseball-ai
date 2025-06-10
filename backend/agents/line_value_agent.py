"""
Line Value Agent - Compares projections to offered lines and identifies +EV opportunities
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
from models.betting.kelly_criterion import KellyCriterion
from data.sources.line_tracker import LineTracker


class LineValueAgent:
    """Agent responsible for identifying value in betting lines"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            openai_api_key=settings.openai_api_key
        )
        self.kelly = KellyCriterion()
        self.line_tracker = LineTracker()
        
        self.tools = [
            self._calculate_implied_probability,
            self._calculate_true_probability,
            self._calculate_edge,
            self._calculate_kelly_bet_size,
            self._track_line_movement,
            self._historical_line_accuracy
        ]
        
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create the LangChain agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a betting value specialist for baseball over/under markets.
            
            Your job is to identify positive expected value (+EV) opportunities by comparing 
            player projections to the lines offered on Underdog Fantasy.
            
            Key responsibilities:
            1. Calculate implied probability from betting lines
            2. Compare to true probability from projections
            3. Calculate expected value and edge
            4. Recommend optimal bet sizing using Kelly Criterion
            5. Track line movement for timing decisions
            
            Only recommend bets with minimum 3% edge and high confidence projections.
            Consider line movement, historical accuracy, and market efficiency.
            
            Always provide clear reasoning for recommendations.
            """),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    @tool
    def _calculate_implied_probability(self, line_value: float, line_type: str = "over") -> Dict:
        """Calculate implied probability from betting line"""
        try:
            # Underdog Fantasy uses -110 standard pricing
            vig_adjusted_prob = 0.5238  # 110/210 with vig
            
            return {
                "implied_probability": vig_adjusted_prob,
                "true_probability": 0.5,  # Before adjusting for projection
                "vig": 0.0476,  # ~4.76% vig
                "line_value": line_value,
                "line_type": line_type
            }
        except Exception as e:
            return {"error": f"Failed to calculate implied probability: {str(e)}"}
    
    @tool
    def _calculate_true_probability(self, projection: float, line_value: float, stat_type: str) -> Dict:
        """Calculate true probability based on projection"""
        try:
            # Use projection distribution to calculate probability
            # Assume normal distribution around projection with stat-specific variance
            
            variance_map = {
                "hits": 0.8,
                "total_bases": 1.2,
                "runs": 0.7,
                "rbis": 0.9,
                "strikeouts": 1.1
            }
            
            std_dev = variance_map.get(stat_type, 1.0)
            
            # Calculate probability of going over the line
            from scipy import stats
            z_score = (line_value - projection) / std_dev
            prob_under = stats.norm.cdf(z_score)
            prob_over = 1 - prob_under
            
            return {
                "true_prob_over": prob_over,
                "true_prob_under": prob_under,
                "projection": projection,
                "line_value": line_value,
                "z_score": z_score,
                "confidence": min(abs(z_score) * 0.2, 1.0)  # Higher confidence for larger deviations
            }
        except Exception as e:
            return {"error": f"Failed to calculate true probability: {str(e)}"}
    
    @tool
    def _calculate_edge(self, true_probability: float, implied_probability: float) -> Dict:
        """Calculate expected value and edge"""
        try:
            # Expected Value = (True Prob * Win Amount) - (Lose Prob * Lose Amount)
            # Standard -110 line: win $100 for $110 bet
            win_amount = 100
            lose_amount = 110
            
            expected_value = (true_probability * win_amount) - ((1 - true_probability) * lose_amount)
            edge = (true_probability - implied_probability) / implied_probability
            
            return {
                "expected_value": expected_value,
                "edge_percentage": edge * 100,
                "true_probability": true_probability,
                "implied_probability": implied_probability,
                "positive_ev": expected_value > 0,
                "meets_threshold": edge >= settings.min_edge_threshold
            }
        except Exception as e:
            return {"error": f"Failed to calculate edge: {str(e)}"}
    
    @tool
    def _calculate_kelly_bet_size(self, edge: float, true_probability: float, bankroll: float) -> Dict:
        """Calculate optimal bet size using Kelly Criterion"""
        try:
            # Kelly formula: f = (bp - q) / b
            # Where: b = odds, p = true prob, q = 1-p
            
            decimal_odds = 1.909  # -110 in decimal
            kelly_fraction = (true_probability * decimal_odds - 1) / (decimal_odds - 1)
            
            # Apply conservative Kelly (25% of full Kelly)
            conservative_fraction = kelly_fraction * settings.kelly_fraction
            
            # Cap at 5% of bankroll for safety
            max_fraction = 0.05
            final_fraction = min(conservative_fraction, max_fraction)
            
            bet_size = final_fraction * bankroll if final_fraction > 0 else 0
            
            return {
                "kelly_fraction": kelly_fraction,
                "conservative_fraction": conservative_fraction,
                "final_fraction": final_fraction,
                "recommended_bet": bet_size,
                "max_bet": max_fraction * bankroll,
                "edge": edge
            }
        except Exception as e:
            return {"error": f"Failed to calculate Kelly bet size: {str(e)}"}
    
    @tool
    def _track_line_movement(self, player_id: int, stat_type: str, game_date: str) -> Dict:
        """Track line movement for timing decisions"""
        try:
            movement = self.line_tracker.get_line_movement(player_id, stat_type, game_date)
            
            return {
                "opening_line": movement.get("opening_line"),
                "current_line": movement.get("current_line"),
                "movement": movement.get("movement", 0),
                "movement_direction": "up" if movement.get("movement", 0) > 0 else "down",
                "volume": movement.get("betting_volume", 0),
                "time_until_game": movement.get("time_until_game"),
                "recommended_timing": "bet_now" if abs(movement.get("movement", 0)) < 0.5 else "wait"
            }
        except Exception as e:
            return {"error": f"Failed to track line movement: {str(e)}"}
    
    @tool
    def _historical_line_accuracy(self, stat_type: str, line_range: Tuple[float, float]) -> Dict:
        """Get historical accuracy for similar lines"""
        try:
            accuracy = self.line_tracker.get_historical_accuracy(stat_type, line_range)
            
            return {
                "historical_over_rate": accuracy.get("over_rate", 0.5),
                "sample_size": accuracy.get("sample_size", 0),
                "market_efficiency": accuracy.get("efficiency_score", 0.5),
                "line_range": line_range,
                "confidence_adjustment": accuracy.get("confidence_factor", 1.0)
            }
        except Exception as e:
            return {"error": f"Failed to get historical accuracy: {str(e)}"}
    
    def analyze_bet_opportunity(self, player_id: int, stat_type: str, line_value: float, 
                              projection: float, confidence: float, bankroll: float) -> Dict:
        """Complete analysis of a betting opportunity"""
        
        query = f"""
        Analyze this betting opportunity:
        - Player ID: {player_id}
        - Stat: {stat_type}
        - Line: {line_value}
        - Projection: {projection}
        - Confidence: {confidence}
        - Bankroll: ${bankroll}
        
        Please:
        1. Calculate implied probability from the line
        2. Calculate true probability from our projection
        3. Calculate expected value and edge
        4. Determine optimal bet size using Kelly Criterion
        5. Check line movement and timing
        6. Review historical accuracy for this line range
        7. Make final recommendation (bet/no bet) with reasoning
        
        Only recommend if edge >= 3% and confidence >= 3/5.
        """
        
        result = self.agent.invoke({"input": query})
        return result["output"]


def analyze_value(player_id: int, stat_type: str, line_value: float, 
                 projection: float, confidence: float, bankroll: float = 1000) -> Dict:
    """Quick value analysis without agent overhead"""
    agent = LineValueAgent()
    return agent.analyze_bet_opportunity(player_id, stat_type, line_value, 
                                       projection, confidence, bankroll)
