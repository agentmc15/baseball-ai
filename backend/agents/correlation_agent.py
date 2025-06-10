"""
Correlation Agent - Identifies correlated outcomes for parlay optimization
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
from models.ml.correlation_analyzer import CorrelationAnalyzer


class CorrelationAgent:
    """Agent responsible for identifying correlated betting outcomes"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            openai_api_key=settings.openai_api_key
        )
        self.correlation_analyzer = CorrelationAnalyzer()
        
        self.tools = [
            self._analyze_player_correlations,
            self._analyze_team_correlations,
            self._analyze_game_correlations,
            self._calculate_parlay_probability,
            self._identify_contrarian_opportunities,
            self._calculate_correlation_matrix
        ]
        
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create the LangChain agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a correlation specialist for baseball betting parlays.
            
            Your expertise is in identifying when multiple betting outcomes are correlated,
            either positively or negatively, to optimize parlay construction.
            
            Key correlation types to analyze:
            1. Player correlations (same player multiple stats)
            2. Team correlations (multiple players from same team)
            3. Game correlations (offensive explosion vs pitcher struggles)
            4. Weather correlations (conditions affecting multiple players)
            5. Situational correlations (lineup changes, injuries)
            
            Goals:
            - Build profitable parlays with correlated outcomes
            - Avoid negative correlations that reduce expected value
            - Identify contrarian opportunities when market misprices correlations
            
            Always consider correlation strength and statistical significance.
            """),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    @tool
    def _analyze_player_correlations(self, player_id: int, stats: List[str]) -> Dict:
        """Analyze correlations between different stats for the same player"""
        try:
            correlations = self.correlation_analyzer.get_player_stat_correlations(player_id, stats)
            
            # Common strong correlations
            strong_correlations = {
                ("hits", "total_bases"): 0.75,
                ("total_bases", "runs"): 0.65,
                ("hits", "runs"): 0.60,
                ("total_bases", "rbis"): 0.70,
                ("hits", "rbis"): 0.55
            }
            
            correlation_pairs = []
            for i, stat1 in enumerate(stats):
                for stat2 in stats[i+1:]:
                    pair = tuple(sorted([stat1, stat2]))
                    correlation = strong_correlations.get(pair, 0.3)  # Default moderate correlation
                    
                    correlation_pairs.append({
                        "stat1": stat1,
                        "stat2": stat2,
                        "correlation": correlation,
                        "strength": "strong" if abs(correlation) > 0.6 else "moderate" if abs(correlation) > 0.3 else "weak",
                        "parlay_value": "positive" if correlation > 0.5 else "neutral"
                    })
            
            return {
                "player_id": player_id,
                "stats_analyzed": stats,
                "correlations": correlation_pairs,
                "best_parlay_combo": max(correlation_pairs, key=lambda x: x["correlation"]) if correlation_pairs else None,
                "recommendation": "favorable_for_parlays" if any(c["correlation"] > 0.6 for c in correlation_pairs) else "neutral"
            }
        except Exception as e:
            return {"error": f"Failed to analyze player correlations: {str(e)}"}
    
    @tool
    def _analyze_team_correlations(self, team_id: int, players: List[int], game_context: Dict) -> Dict:
        """Analyze correlations between players on the same team"""
        try:
            team_correlations = self.correlation_analyzer.get_team_correlations(team_id, players)
            
            # Team offensive correlations
            # When team has a good offensive game, multiple players benefit
            
            lineup_position_correlations = {
                "leadoff_hits_team_runs": 0.4,
                "cleanup_rbis_team_runs": 0.6,
                "multiple_hits_same_inning": 0.3,
                "pitcher_struggles_multiple_hitters": 0.5
            }
            
            weather_correlations = game_context.get("weather_correlations", {})
            opposing_pitcher = game_context.get("opposing_pitcher_weakness", False)
            
            # Calculate team offensive explosion probability
            explosion_factors = []
            if weather_correlations.get("favorable_hitting_conditions"):
                explosion_factors.append({"factor": "weather", "boost": 0.15})
            if opposing_pitcher:
                explosion_factors.append({"factor": "weak_pitcher", "boost": 0.20})
            
            total_boost = sum(f["boost"] for f in explosion_factors)
            
            return {
                "team_id": team_id,
                "players_analyzed": players,
                "lineup_correlations": lineup_position_correlations,
                "explosion_probability": min(0.3 + total_boost, 0.6),  # Cap at 60%
                "explosion_factors": explosion_factors,
                "parlay_recommendation": "favorable" if total_boost > 0.15 else "neutral",
                "best_correlation_type": "offensive_explosion" if total_boost > 0.2 else "individual_performance"
            }
        except Exception as e:
            return {"error": f"Failed to analyze team correlations: {str(e)}"}
    
    @tool
    def _analyze_game_correlations(self, game_id: str, total_runs_line: float) -> Dict:
        """Analyze correlations within a specific game"""
        try:
            # Game-level correlations
            correlations = {
                "high_scoring_game": {
                    "individual_overs": 0.4,   # Individual player overs more likely
                    "pitcher_unders": 0.3,     # Pitcher strikeout unders more likely
                    "total_bases_overs": 0.5   # Total bases strongly correlated
                },
                "pitcher_duel": {
                    "individual_unders": 0.35,
                    "pitcher_overs": 0.4,      # Strikeout overs more likely
                    "runs_unders": 0.6
                },
                "weather_game": {
                    "offensive_overs": 0.3,
                    "home_run_overs": 0.5,
                    "total_bases_correlation": 0.4
                }
            }
            
            # Determine game script probability
            if total_runs_line > 9.5:
                game_script = "high_scoring_game"
            elif total_runs_line < 7.5:
                game_script = "pitcher_duel"
            else:
                game_script = "weather_game"  # Default to weather-dependent
            
            relevant_correlations = correlations[game_script]
            
            return {
                "game_id": game_id,
                "total_runs_line": total_runs_line,
                "predicted_game_script": game_script,
                "correlations": relevant_correlations,
                "parlay_strategy": f"Build parlays around {game_script.replace('_', ' ')} theme",
                "confidence": 0.7 if abs(total_runs_line - 8.5) > 1 else 0.5
            }
        except Exception as e:
            return {"error": f"Failed to analyze game correlations: {str(e)}"}
    
    @tool
    def _calculate_parlay_probability(self, individual_bets: List[Dict], correlations: Dict) -> Dict:
        """Calculate true parlay probability accounting for correlations"""
        try:
            # Individual probabilities
            individual_probs = [bet["probability"] for bet in individual_bets]
            
            # Naive independent probability
            independent_prob = np.prod(individual_probs)
            
            # Adjust for correlations
            correlation_adjustment = 1.0
            
            for i, bet1 in enumerate(individual_bets):
                for j, bet2 in enumerate(individual_bets[i+1:], i+1):
                    correlation_key = f"{bet1['stat']}_{bet2['stat']}"
                    correlation = correlations.get(correlation_key, 0)
                    
                    # Positive correlation increases parlay probability
                    # Negative correlation decreases it
                    if correlation > 0:
                        correlation_adjustment += correlation * 0.1  # Max 10% boost per correlation
                    else:
                        correlation_adjustment += correlation * 0.05  # Max 5% penalty per correlation
            
            adjusted_prob = independent_prob * correlation_adjustment
            adjusted_prob = max(0.01, min(0.99, adjusted_prob))  # Bound between 1% and 99%
            
            # Calculate expected value
            parlay_odds = 1 / independent_prob  # Bookmaker uses independent assumption
            parlay_payout = parlay_odds - 1
            expected_value = (adjusted_prob * parlay_payout) - ((1 - adjusted_prob) * 1)
            
            return {
                "individual_probabilities": individual_probs,
                "independent_probability": independent_prob,
                "correlation_adjustment": correlation_adjustment,
                "adjusted_probability": adjusted_prob,
                "parlay_odds": parlay_odds,
                "expected_value": expected_value,
                "positive_ev": expected_value > 0,
                "edge": (adjusted_prob - (1/parlay_odds)) * 100
            }
        except Exception as e:
            return {"error": f"Failed to calculate parlay probability: {str(e)}"}
    
    @tool
    def _identify_contrarian_opportunities(self, market_data: Dict, correlation_analysis: Dict) -> Dict:
        """Identify contrarian opportunities where market misprices correlations"""
        try:
            # Look for situations where public is wrong about correlations
            contrarian_signals = []
            
            # High public betting on uncorrelated outcomes
            if market_data.get("public_betting_percentage", 0) > 70:
                contrarian_signals.append({
                    "type": "fade_public",
                    "reasoning": "Public overvaluing low-correlation parlay",
                    "opportunity": "bet_against"
                })
            
            # Market undervaluing positive correlations
            strong_correlations = [c for c in correlation_analysis.get("correlations", []) 
                                 if c.get("correlation", 0) > 0.6]
            
            if strong_correlations and market_data.get("parlay_odds", 0) > correlation_analysis.get("fair_odds", 0) * 1.1:
                contrarian_signals.append({
                    "type": "undervalued_correlation",
                    "reasoning": "Market not properly pricing positive correlation",
                    "opportunity": "bet_parlay"
                })
            
            # Weather-based correlations market is missing
            if correlation_analysis.get("weather_factor", 0) > 0.15:
                contrarian_signals.append({
                    "type": "weather_correlation",
                    "reasoning": "Market slow to adjust for weather correlations",
                    "opportunity": "weather_based_parlay"
                })
            
            return {
                "contrarian_signals": contrarian_signals,
                "opportunity_count": len(contrarian_signals),
                "highest_value_signal": max(contrarian_signals, key=lambda x: 1) if contrarian_signals else None,
                "recommendation": "contrarian_bet" if len(contrarian_signals) >= 2 else "standard_analysis"
            }
        except Exception as e:
            return {"error": f"Failed to identify contrarian opportunities: {str(e)}"}
    
    @tool
    def _calculate_correlation_matrix(self, players: List[int], stats: List[str], game_context: Dict) -> Dict:
        """Calculate full correlation matrix for complex parlay analysis"""
        try:
            # Build correlation matrix
            matrix = {}
            
            for player1 in players:
                for stat1 in stats:
                    key1 = f"player_{player1}_{stat1}"
                    matrix[key1] = {}
                    
                    for player2 in players:
                        for stat2 in stats:
                            key2 = f"player_{player2}_{stat2}"
                            
                            if player1 == player2:
                                # Same player correlations
                                correlation = self._get_stat_correlation(stat1, stat2)
                            elif game_context.get("same_team", False):
                                # Teammates correlation
                                correlation = 0.2  # Moderate positive
                            else:
                                # Opponents correlation
                                correlation = -0.1  # Slight negative
                            
                            matrix[key1][key2] = correlation
            
            # Find best 2-leg, 3-leg, and 4-leg parlays
            best_parlays = self._find_optimal_parlays(matrix, 2)
            
            return {
                "correlation_matrix": matrix,
                "players": players,
                "stats": stats,
                "best_2_leg_parlay": best_parlays.get("2_leg"),
                "best_3_leg_parlay": best_parlays.get("3_leg"),
                "matrix_size": len(matrix),
                "average_correlation": np.mean([v for row in matrix.values() for v in row.values()])
            }
        except Exception as e:
            return {"error": f"Failed to calculate correlation matrix: {str(e)}"}
    
    def _get_stat_correlation(self, stat1: str, stat2: str) -> float:
        """Get correlation between two stats"""
        correlations = {
            ("hits", "total_bases"): 0.75,
            ("hits", "runs"): 0.60,
            ("hits", "rbis"): 0.55,
            ("total_bases", "runs"): 0.65,
            ("total_bases", "rbis"): 0.70,
            ("runs", "rbis"): 0.45
        }
        
        pair = tuple(sorted([stat1, stat2]))
        return correlations.get(pair, 0.0 if stat1 != stat2 else 1.0)
    
    def _find_optimal_parlays(self, matrix: Dict, max_legs: int) -> Dict:
        """Find optimal parlay combinations"""
        # Simplified implementation - would use more complex optimization in production
        return {
            "2_leg": {"combination": ["player_1_hits", "player_1_total_bases"], "correlation": 0.75},
            "3_leg": {"combination": ["player_1_hits", "player_1_total_bases", "player_2_hits"], "correlation": 0.45}
        }
    
    def analyze_parlay_opportunity(self, players: List[int], stats: List[str], 
                                 game_context: Dict, market_data: Dict) -> Dict:
        """Complete parlay correlation analysis"""
        
        query = f"""
        Analyze parlay opportunities for:
        - Players: {players}
        - Stats: {stats}
        - Game context: {game_context}
        - Market data: {market_data}
        
        Please:
        1. Analyze individual player stat correlations
        2. Analyze team-level correlations if applicable
        3. Analyze game-level correlations
        4. Calculate true parlay probabilities with correlation adjustments
        5. Identify any contrarian opportunities
        6. Build correlation matrix for optimal parlay construction
        7. Recommend best parlay combinations with reasoning
        
        Focus on finding positive expected value parlays using correlation insights.
        """
        
        result = self.agent.invoke({"input": query})
        return result["output"]


def quick_correlation_analysis(players: List[int], stats: List[str]) -> Dict:
    """Quick correlation analysis for simple parlays"""
    agent = CorrelationAgent()
    return agent.analyze_parlay_opportunity(players, stats, {}, {})
