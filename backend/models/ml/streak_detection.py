"""
Hidden Markov Model for hot/cold streak detection
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import joblib
import os

from utils.config import settings


class StreakDetector:
    """Hidden Markov Model for detecting player performance states"""
    
    def __init__(self):
        self.models = {}  # Models for each stat type
        self.scalers = {}  # Feature scalers
        self.model_path = "backend/models/artifacts"
        
        # Performance states
        self.states = ["Cold", "Average", "Hot"]
        self.n_states = len(self.states)
        
        # Load pre-trained models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained HMM models"""
        for stat in ["hits", "total_bases", "runs", "rbis"]:
            model_file = f"{self.model_path}/hmm_{stat}_model.joblib"
            scaler_file = f"{self.model_path}/hmm_{stat}_scaler.joblib"
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                self.models[stat] = joblib.load(model_file)
                self.scalers[stat] = joblib.load(scaler_file)
                print(f"Loaded HMM model for {stat}")
    
    def prepare_features(self, game_logs: List[Dict], stat_type: str) -> np.ndarray:
        """
        Prepare features for HMM from game logs
        Features include: stat value, moving averages, recent trend
        """
        if not game_logs:
            return np.array([]).reshape(0, 4)
        
        df = pd.DataFrame(game_logs)
        stat_values = df[stat_type].values
        
        features = []
        for i, value in enumerate(stat_values):
            # Current stat value
            current_stat = value
            
            # 3-game moving average
            start_idx = max(0, i - 2)
            ma_3 = np.mean(stat_values[start_idx:i+1])
            
            # 7-game moving average
            start_idx = max(0, i - 6)
            ma_7 = np.mean(stat_values[start_idx:i+1])
            
            # Recent trend (last 3 vs previous 3)
            if i >= 5:
                recent_3 = np.mean(stat_values[i-2:i+1])
                prev_3 = np.mean(stat_values[i-5:i-2])
                trend = recent_3 - prev_3
            else:
                trend = 0
            
            features.append([current_stat, ma_3, ma_7, trend])
        
        return np.array(features)
    
    def train_hmm_model(self, training_data: List[List[Dict]], stat_type: str):
        """
        Train HMM model on multiple players' game logs
        training_data: List of player game logs
        """
        all_features = []
        all_lengths = []
        
        # Prepare training data
        for player_logs in training_data:
            features = self.prepare_features(player_logs, stat_type)
            if len(features) > 5:  # Minimum sequence length
                all_features.append(features)
                all_lengths.append(len(features))
        
        if not all_features:
            print(f"No sufficient data for {stat_type} HMM training")
            return
        
        # Concatenate all features
        X = np.vstack(all_features)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train HMM
        model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        try:
            model.fit(X_scaled, lengths=all_lengths)
            
            # Save model and scaler
            os.makedirs(self.model_path, exist_ok=True)
            joblib.dump(model, f"{self.model_path}/hmm_{stat_type}_model.joblib")
            joblib.dump(scaler, f"{self.model_path}/hmm_{stat_type}_scaler.joblib")
            
            self.models[stat_type] = model
            self.scalers[stat_type] = scaler
            
            print(f"Trained and saved HMM model for {stat_type}")
            
        except Exception as e:
            print(f"Error training HMM for {stat_type}: {e}")
    
    def detect_current_state(self, recent_games: List[Dict], stat_type: str) -> Dict:
        """
        Detect current performance state for a player
        """
        if stat_type not in self.models or stat_type not in self.scalers:
            # Fallback to simple heuristic
            return self._simple_streak_detection(recent_games, stat_type)
        
        # Prepare features
        features = self.prepare_features(recent_games, stat_type)
        if len(features) == 0:
            return {"state": "Average", "confidence": 0.0, "probability": [0.33, 0.34, 0.33]}
        
        # Scale features
        features_scaled = self.scalers[stat_type].transform(features)
        
        # Predict states
        model = self.models[stat_type]
        
        try:
            # Get state probabilities for the sequence
            log_prob, state_sequence = model.decode(features_scaled)
            state_probabilities = model.predict_proba(features_scaled)
            
            # Current state is the last predicted state
            current_state_idx = state_sequence[-1]
            current_state = self.states[current_state_idx]
            
            # Confidence is the probability of the current state
            current_state_prob = state_probabilities[-1, current_state_idx]
            
            # Calculate streak length
            streak_length = self._calculate_streak_length(state_sequence, current_state_idx)
            
            return {
                "state": current_state,
                "confidence": current_state_prob,
                "probability": state_probabilities[-1].tolist(),
                "streak_length": streak_length,
                "log_likelihood": log_prob / len(features_scaled),
                "state_sequence": [self.states[s] for s in state_sequence[-5:]]  # Last 5 games
            }
            
        except Exception as e:
            print(f"Error in HMM prediction for {stat_type}: {e}")
            return self._simple_streak_detection(recent_games, stat_type)
    
    def _simple_streak_detection(self, recent_games: List[Dict], stat_type: str) -> Dict:
        """
        Fallback simple streak detection based on recent performance
        """
        if len(recent_games) < 3:
            return {"state": "Average", "confidence": 0.5, "probability": [0.33, 0.34, 0.33]}
        
        df = pd.DataFrame(recent_games)
        stat_values = df[stat_type].values
        
        # Calculate recent average vs season average (estimate)
        recent_avg = np.mean(stat_values[-5:]) if len(stat_values) >= 5 else np.mean(stat_values)
        season_avg = np.mean(stat_values)  # Approximate with available data
        
        # Calculate z-score
        std_dev = np.std(stat_values) if np.std(stat_values) > 0 else 1.0
        z_score = (recent_avg - season_avg) / std_dev
        
        # Classify state
        if z_score > 0.5:
            state = "Hot"
            confidence = min(0.8, 0.5 + abs(z_score) * 0.1)
        elif z_score < -0.5:
            state = "Cold"
            confidence = min(0.8, 0.5 + abs(z_score) * 0.1)
        else:
            state = "Average"
            confidence = 0.6
        
        # Create probability distribution
        if state == "Hot":
            probs = [0.1, 0.3, 0.6]
        elif state == "Cold":
            probs = [0.6, 0.3, 0.1]
        else:
            probs = [0.25, 0.5, 0.25]
        
        return {
            "state": state,
            "confidence": confidence,
            "probability": probs,
            "z_score": z_score,
            "method": "simple_heuristic"
        }
    
    def _calculate_streak_length(self, state_sequence: np.ndarray, current_state: int) -> int:
        """Calculate how long the player has been in current state"""
        streak_length = 1
        
        # Count backwards from the end
        for i in range(len(state_sequence) - 2, -1, -1):
            if state_sequence[i] == current_state:
                streak_length += 1
            else:
                break
        
        return streak_length
    
    def calculate_streak_adjustment(self, streak_info: Dict, stat_type: str) -> float:
        """
        Calculate adjustment factor based on current streak
        """
        state = streak_info.get("state", "Average")
        confidence = streak_info.get("confidence", 0.5)
        streak_length = streak_info.get("streak_length", 1)
        
        # Base adjustments by state
        base_adjustments = {
            "Hot": 1.15,    # 15% boost
            "Average": 1.0,  # No adjustment
            "Cold": 0.85    # 15% penalty
        }
        
        base_adj = base_adjustments[state]
        
        # Streak length effect (longer streaks have stronger effect)
        length_multiplier = min(1 + (streak_length - 1) * 0.05, 1.3)  # Cap at 30% additional
        
        # Confidence effect (lower confidence reduces adjustment)
        confidence_multiplier = 0.5 + (confidence * 0.5)  # Scale confidence effect
        
        # Calculate final adjustment
        final_adjustment = 1 + (base_adj - 1) * length_multiplier * confidence_multiplier
        
        return max(0.7, min(1.4, final_adjustment))  # Cap between 70% and 140%
    
    def analyze_player_streak(self, player_id: int, recent_games: List[Dict], 
                             stats: List[str] = None) -> Dict:
        """
        Complete streak analysis for a player across multiple stats
        """
        if stats is None:
            stats = ["hits", "total_bases", "runs", "rbis"]
        
        analysis = {
            "player_id": player_id,
            "analysis_date": pd.Timestamp.now().isoformat(),
            "games_analyzed": len(recent_games),
            "stats": {}
        }
        
        for stat in stats:
            streak_info = self.detect_current_state(recent_games, stat)
            adjustment = self.calculate_streak_adjustment(streak_info, stat)
            
            analysis["stats"][stat] = {
                "current_state": streak_info["state"],
                "confidence": streak_info["confidence"],
                "streak_length": streak_info.get("streak_length", 1),
                "adjustment_factor": adjustment,
                "state_probabilities": streak_info["probability"]
            }
        
        # Overall player state (majority vote)
        states = [info["current_state"] for info in analysis["stats"].values()]
        overall_state = max(set(states), key=states.count)
        
        analysis["overall_state"] = overall_state
        analysis["state_consistency"] = states.count(overall_state) / len(states)
        
        return analysis


def quick_streak_analysis(recent_games: List[Dict], stat_type: str = "hits") -> Dict:
    """Quick streak analysis for a single stat"""
    detector = StreakDetector()
    return detector.detect_current_state(recent_games, stat_type)
