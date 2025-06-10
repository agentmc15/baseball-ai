"""
Base ML projection model using XGBoost
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

from utils.config import settings


class BaseProjectionModel:
    """XGBoost-based player projection model"""
    
    def __init__(self):
        self.models = {}  # Separate model for each stat
        self.feature_names = []
        self.model_path = "backend/models/artifacts"
        
        # Stats to predict
        self.stats = ["hits", "total_bases", "runs", "rbis", "strikeouts"]
        
        # Load pre-trained models if they exist
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        for stat in self.stats:
            model_file = f"{self.model_path}/{stat}_model.joblib"
            if os.path.exists(model_file):
                self.models[stat] = joblib.load(model_file)
                print(f"Loaded {stat} model")
    
    def prepare_features(self, player_data: Dict) -> np.ndarray:
        """Prepare features for prediction"""
        features = []
        
        # Recent performance features (last 15 games)
        recent_stats = player_data.get("recent_stats", [])
        if recent_stats:
            recent_df = pd.DataFrame(recent_stats)
            
            # Rolling averages
            features.extend([
                recent_df["hits"].mean(),
                recent_df["total_bases"].mean(),
                recent_df["runs"].mean(),
                recent_df["rbis"].mean(),
                recent_df["strikeouts"].mean(),
                recent_df["at_bats"].mean()
            ])
            
            # Recent trends (last 5 vs previous 10)
            if len(recent_stats) >= 10:
                last_5 = recent_df.tail(5)
                prev_10 = recent_df.head(10)
                
                features.extend([
                    last_5["hits"].mean() - prev_10["hits"].mean(),
                    last_5["total_bases"].mean() - prev_10["total_bases"].mean(),
                    last_5["runs"].mean() - prev_10["runs"].mean()
                ])
            else:
                features.extend([0, 0, 0])  # No trend data
            
            # Consistency metrics
            features.extend([
                recent_df["hits"].std(),
                recent_df["total_bases"].std(),
                recent_df["runs"].std()
            ])
        else:
            # No recent data - use defaults
            features.extend([0] * 12)
        
        # Opponent strength features
        features.extend([
            player_data.get("opposing_pitcher_era", 4.50),
            player_data.get("opposing_pitcher_whip", 1.30),
            player_data.get("opposing_team_def_rating", 0.0)
        ])
        
        # Situational features
        features.extend([
            player_data.get("home_game", 0),  # 1 if home, 0 if away
            player_data.get("day_game", 0),   # 1 if day, 0 if night
            player_data.get("rest_days", 0)   # Days of rest
        ])
        
        self.feature_names = [
            "avg_hits", "avg_total_bases", "avg_runs", "avg_rbis", "avg_strikeouts", "avg_at_bats",
            "trend_hits", "trend_total_bases", "trend_runs",
            "std_hits", "std_total_bases", "std_runs",
            "opp_pitcher_era", "opp_pitcher_whip", "opp_team_def",
            "home_game", "day_game", "rest_days"
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, player_data: Dict) -> Dict:
        """Generate predictions for all stats"""
        features = self.prepare_features(player_data)
        predictions = {}
        
        for stat in self.stats:
            if stat in self.models:
                # Use trained model
                pred = self.models[stat].predict(features)[0]
                
                # Get prediction intervals
                if hasattr(self.models[stat], 'predict_proba'):
                    # For models that support it
                    confidence = 0.75
                else:
                    # Estimate confidence based on recent performance variance
                    recent_stats = player_data.get("recent_stats", [])
                    if recent_stats and len(recent_stats) >= 5:
                        recent_values = [game.get(stat, 0) for game in recent_stats[-5:]]
                        variance = np.var(recent_values)
                        confidence = max(0.3, min(0.9, 1.0 - variance / 2.0))
                    else:
                        confidence = 0.5
                
                predictions[stat] = {
                    "prediction": max(0, pred),  # Ensure non-negative
                    "confidence": confidence,
                    "lower_bound": max(0, pred - 1.0),
                    "upper_bound": pred + 1.0
                }
            else:
                # Fallback to simple average if no model
                recent_stats = player_data.get("recent_stats", [])
                if recent_stats:
                    avg_value = np.mean([game.get(stat, 0) for game in recent_stats])
                else:
                    avg_value = {"hits": 1.0, "total_bases": 1.5, "runs": 0.8, "rbis": 0.9, "strikeouts": 1.2}[stat]
                
                predictions[stat] = {
                    "prediction": avg_value,
                    "confidence": 0.4,  # Low confidence for fallback
                    "lower_bound": avg_value - 0.5,
                    "upper_bound": avg_value + 0.5
                }
        
        return predictions
    
    def train_model(self, training_data: pd.DataFrame, stat: str):
        """Train XGBoost model for a specific stat"""
        # Prepare features and target
        feature_columns = [col for col in training_data.columns if col not in self.stats]
        X = training_data[feature_columns]
        y = training_data[stat]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            objective='reg:squarederror'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"{stat} model - MAE: {mae:.3f}, RMSE: {rmse:.3f}")
        
        # Save model
        self.models[stat] = model
        os.makedirs(self.model_path, exist_ok=True)
        joblib.dump(model, f"{self.model_path}/{stat}_model.joblib")
        
        return {"mae": mae, "rmse": rmse}
