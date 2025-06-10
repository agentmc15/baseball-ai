"""
Predictions API endpoints using agent orchestration
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import date

from graphs.prediction_graph import BaseballPredictionGraph, quick_prediction
from agents.projection_agent import quick_projection
from agents.line_value_agent import analyze_value

router = APIRouter()

# Pydantic models
class PredictionRequest(BaseModel):
    player_id: int
    game_id: str
    stat_type: str
    line_value: Optional[float] = None
    bankroll: Optional[float] = 1000

class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest]

class PredictionResponse(BaseModel):
    player_id: int
    game_id: str
    stat_type: str
    projection: dict
    recommendation: dict
    confidence: float
    reasoning: List[str]

@router.get("/")
async def get_predictions_info():
    """Get information about prediction endpoints"""
    return {
        "message": "Baseball AI Predictions API",
        "endpoints": {
            "/predict": "Generate single prediction",
            "/batch": "Generate multiple predictions",
            "/quick/{player_id}": "Quick projection without line analysis",
            "/daily-slate": "Get predictions for today's slate"
        }
    }

@router.post("/predict", response_model=PredictionResponse)
async def generate_prediction(request: PredictionRequest):
    """Generate complete prediction with recommendation"""
    try:
        graph = BaseballPredictionGraph()
        result = graph.run_prediction(
            request.player_id,
            request.game_id,
            request.stat_type,
            request.line_value,
            request.bankroll
        )
        
        return PredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/batch")
async def generate_batch_predictions(request: BatchPredictionRequest):
    """Generate multiple predictions"""
    results = []
    
    for pred_request in request.predictions:
        try:
            result = quick_prediction(
                pred_request.player_id,
                pred_request.game_id,
                pred_request.stat_type,
                pred_request.line_value,
                pred_request.bankroll
            )
            results.append(result)
        except Exception as e:
            results.append({
                "player_id": pred_request.player_id,
                "error": str(e)
            })
    
    return {"predictions": results, "total": len(results)}

@router.get("/quick/{player_id}")
async def quick_player_projection(
    player_id: int,
    game_id: str = Query(..., description="Game ID"),
    stat_type: str = Query(default="hits", description="Stat type to project")
):
    """Quick projection without full analysis"""
    try:
        projection = quick_projection(player_id, game_id, stat_type)
        return {
            "player_id": player_id,
            "game_id": game_id,
            "stat_type": stat_type,
            "projection": projection
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick projection failed: {str(e)}")

@router.get("/daily-slate")
async def get_daily_slate(
    game_date: Optional[str] = Query(default=None, description="Date in YYYY-MM-DD format")
):
    """Get predictions for today's slate (placeholder)"""
    if not game_date:
        game_date = date.today().isoformat()
    
    # Placeholder response - would integrate with MLB API for real games
    return {
        "date": game_date,
        "games": [
            {
                "game_id": "662883",
                "home_team": "Yankees",
                "away_team": "Red Sox",
                "predictions": [
                    {
                        "player_id": 545361,
                        "player_name": "Aaron Judge",
                        "stat_type": "total_bases",
                        "projection": 2.1,
                        "line": 1.5,
                        "recommendation": "OVER",
                        "confidence": 0.75,
                        "edge": 8.5
                    }
                ]
            }
        ],
        "total_opportunities": 1,
        "high_confidence_picks": 1
    }

@router.get("/analyze-value")
async def analyze_betting_value(
    player_id: int = Query(..., description="Player ID"),
    stat_type: str = Query(..., description="Stat type"),
    line_value: float = Query(..., description="Betting line"),
    projection: float = Query(..., description="Projected value"),
    confidence: float = Query(default=0.7, description="Confidence level"),
    bankroll: float = Query(default=1000, description="Available bankroll")
):
    """Analyze betting value for specific opportunity"""
    try:
        analysis = analyze_value(
            player_id, stat_type, line_value, 
            projection, confidence, bankroll
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Value analysis failed: {str(e)}")

@router.get("/player/{player_id}/recent-form")
async def get_player_recent_form(player_id: int, days: int = Query(default=15)):
    """Get player's recent performance form"""
    # Placeholder - would integrate with MLB data source
    return {
        "player_id": player_id,
        "days_analyzed": days,
        "recent_stats": {
            "hits": 1.2,
            "total_bases": 1.8,
            "runs": 0.9,
            "rbis": 1.1,
            "strikeouts": 1.3
        },
        "trending": "up",
        "hot_streak": False,
        "games_played": 12
    }
