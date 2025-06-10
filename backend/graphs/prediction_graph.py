"""
LangGraph orchestration for baseball prediction workflow
"""
from typing import Dict, List, Optional, TypedDict
from langgraph import StateGraph, END
from langchain_openai import ChatOpenAI

from agents.projection_agent import ProjectionAgent
from agents.line_value_agent import LineValueAgent
from agents.weather_agent import WeatherAgent
from agents.correlation_agent import CorrelationAgent
from utils.config import settings


class PredictionState(TypedDict):
    """State object for prediction workflow"""
    player_id: int
    game_id: str
    stat_type: str
    line_value: Optional[float]
    bankroll: float
    
    # Agent outputs
    projection: Optional[Dict]
    weather_analysis: Optional[Dict]
    line_analysis: Optional[Dict]
    correlation_analysis: Optional[Dict]
    
    # Final outputs
    recommendation: Optional[Dict]
    confidence: Optional[float]
    reasoning: List[str]


class BaseballPredictionGraph:
    """LangGraph workflow for baseball predictions"""
    
    def __init__(self):
        self.projection_agent = ProjectionAgent()
        self.line_value_agent = LineValueAgent()
        self.weather_agent = WeatherAgent()
        self.correlation_agent = CorrelationAgent()
        
        # Create the graph
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the prediction workflow graph"""
        workflow = StateGraph(PredictionState)
        
        # Add nodes
        workflow.add_node("generate_projection", self._generate_projection)
        workflow.add_node("analyze_weather", self._analyze_weather)
        workflow.add_node("analyze_line_value", self._analyze_line_value)
        workflow.add_node("analyze_correlations", self._analyze_correlations)
        workflow.add_node("make_final_recommendation", self._make_final_recommendation)
        
        # Add edges
        workflow.set_entry_point("generate_projection")
        workflow.add_edge("generate_projection", "analyze_weather")
        workflow.add_edge("analyze_weather", "analyze_line_value")
        workflow.add_edge("analyze_line_value", "analyze_correlations")
        workflow.add_edge("analyze_correlations", "make_final_recommendation")
        workflow.add_edge("make_final_recommendation", END)
        
        return workflow.compile()
    
    def _generate_projection(self, state: PredictionState) -> PredictionState:
        """Generate base projection for the player"""
        try:
            projection = self.projection_agent.generate_projection(
                state["player_id"],
                state["game_id"],
                state["stat_type"]
            )
            state["projection"] = projection
            state["reasoning"].append(f"Generated projection: {projection}")
        except Exception as e:
            state["reasoning"].append(f"Projection failed: {str(e)}")
            state["projection"] = None
        
        return state
    
    def _analyze_weather(self, state: PredictionState) -> PredictionState:
        """Analyze weather impact"""
        try:
            # Extract venue from game_id (simplified)
            venue_id = 1  # Would extract from game data
            game_time = "2024-07-15T19:10:00Z"  # Would get from game data
            
            weather_analysis = self.weather_agent.analyze_weather_impact(
                venue_id,
                game_time,
                state["player_id"],
                state["stat_type"]
            )
            state["weather_analysis"] = weather_analysis
            state["reasoning"].append(f"Weather analysis: {weather_analysis}")
        except Exception as e:
            state["reasoning"].append(f"Weather analysis failed: {str(e)}")
            state["weather_analysis"] = None
        
        return state
    
    def _analyze_line_value(self, state: PredictionState) -> PredictionState:
        """Analyze betting line value"""
        if not state["line_value"] or not state["projection"]:
            state["reasoning"].append("Skipping line analysis - missing data")
            return state
        
        try:
            line_analysis = self.line_value_agent.analyze_bet_opportunity(
                state["player_id"],
                state["stat_type"],
                state["line_value"],
                state["projection"].get("prediction", 0),
                state["projection"].get("confidence", 0.5),
                state["bankroll"]
            )
            state["line_analysis"] = line_analysis
            state["reasoning"].append(f"Line analysis: {line_analysis}")
        except Exception as e:
            state["reasoning"].append(f"Line analysis failed: {str(e)}")
            state["line_analysis"] = None
        
        return state
    
    def _analyze_correlations(self, state: PredictionState) -> PredictionState:
        """Analyze correlations for parlay opportunities"""
        try:
            # Simple correlation analysis for now
            correlation_analysis = self.correlation_agent.analyze_parlay_opportunity(
                [state["player_id"]],
                [state["stat_type"]],
                {"game_id": state["game_id"]},
                {"line_value": state["line_value"]}
            )
            state["correlation_analysis"] = correlation_analysis
            state["reasoning"].append(f"Correlation analysis: {correlation_analysis}")
        except Exception as e:
            state["reasoning"].append(f"Correlation analysis failed: {str(e)}")
            state["correlation_analysis"] = None
        
        return state
    
    def _make_final_recommendation(self, state: PredictionState) -> PredictionState:
        """Make final betting recommendation"""
        try:
            # Combine all analyses
            projection = state.get("projection", {})
            weather = state.get("weather_analysis", {})
            line_analysis = state.get("line_analysis", {})
            
            # Calculate final confidence
            base_confidence = projection.get("confidence", 0.5)
            weather_confidence = weather.get("confidence", 0.5) if weather else 0.5
            
            final_confidence = (base_confidence + weather_confidence) / 2
            
            # Make recommendation
            if line_analysis and line_analysis.get("positive_ev", False):
                recommendation = {
                    "action": "BET",
                    "bet_type": "OVER" if projection.get("prediction", 0) > state.get("line_value", 0) else "UNDER",
                    "confidence": final_confidence,
                    "recommended_bet_size": line_analysis.get("recommended_bet", 0),
                    "expected_value": line_analysis.get("expected_value", 0),
                    "edge": line_analysis.get("edge", 0)
                }
            else:
                recommendation = {
                    "action": "NO BET",
                    "reason": "Insufficient edge or negative expected value",
                    "confidence": final_confidence
                }
            
            state["recommendation"] = recommendation
            state["confidence"] = final_confidence
            state["reasoning"].append(f"Final recommendation: {recommendation}")
            
        except Exception as e:
            state["reasoning"].append(f"Final recommendation failed: {str(e)}")
            state["recommendation"] = {"action": "ERROR", "reason": str(e)}
        
        return state
    
    def run_prediction(self, player_id: int, game_id: str, stat_type: str, 
                      line_value: float = None, bankroll: float = 1000) -> Dict:
        """Run complete prediction workflow"""
        
        initial_state = PredictionState(
            player_id=player_id,
            game_id=game_id,
            stat_type=stat_type,
            line_value=line_value,
            bankroll=bankroll,
            projection=None,
            weather_analysis=None,
            line_analysis=None,
            correlation_analysis=None,
            recommendation=None,
            confidence=None,
            reasoning=[]
        )
        
        # Run the workflow
        final_state = self.graph.invoke(initial_state)
        
        return {
            "player_id": player_id,
            "game_id": game_id,
            "stat_type": stat_type,
            "line_value": line_value,
            "projection": final_state.get("projection"),
            "weather_analysis": final_state.get("weather_analysis"),
            "line_analysis": final_state.get("line_analysis"),
            "recommendation": final_state.get("recommendation"),
            "confidence": final_state.get("confidence"),
            "reasoning": final_state.get("reasoning", [])
        }


# Quick prediction function
def quick_prediction(player_id: int, game_id: str, stat_type: str, 
                    line_value: float = None, bankroll: float = 1000) -> Dict:
    """Quick prediction using the graph workflow"""
    graph = BaseballPredictionGraph()
    return graph.run_prediction(player_id, game_id, stat_type, line_value, bankroll)
