"""
Kelly Criterion implementation for optimal bet sizing
"""
from typing import Dict, Optional
import numpy as np

from utils.config import settings


class KellyCriterion:
    """Kelly Criterion calculator for baseball betting"""
    
    def __init__(self):
        self.max_fraction = settings.kelly_fraction  # Conservative fraction (default 25%)
        self.min_edge = settings.min_edge_threshold   # Minimum edge to bet (default 3%)
        self.max_bet_size = 0.05  # Maximum 5% of bankroll per bet
    
    def calculate_kelly_fraction(self, true_probability: float, decimal_odds: float) -> float:
        """
        Calculate Kelly fraction: f = (bp - q) / b
        where: b = decimal odds - 1, p = true probability, q = 1 - p
        """
        if true_probability <= 0 or true_probability >= 1:
            return 0.0
        
        b = decimal_odds - 1  # Net odds
        p = true_probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        return max(0, kelly_fraction)  # Never bet negative Kelly
    
    def calculate_edge(self, true_probability: float, implied_probability: float) -> float:
        """Calculate betting edge as percentage"""
        if implied_probability <= 0:
            return 0.0
        
        edge = (true_probability - implied_probability) / implied_probability
        return edge
    
    def american_to_decimal_odds(self, american_odds: int) -> float:
        """Convert American odds to decimal odds"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    def calculate_optimal_bet_size(self, true_probability: float, american_odds: int, 
                                 bankroll: float, confidence: float = 1.0) -> Dict:
        """
        Calculate optimal bet size using Kelly Criterion
        """
        # Convert odds
        decimal_odds = self.american_to_decimal_odds(american_odds)
        implied_probability = 1 / decimal_odds
        
        # Calculate edge
        edge = self.calculate_edge(true_probability, implied_probability)
        
        # Only bet if edge meets minimum threshold
        if edge < self.min_edge:
            return {
                "recommended_bet": 0,
                "kelly_fraction": 0,
                "edge": edge * 100,
                "reason": f"Edge {edge*100:.1f}% below minimum threshold {self.min_edge*100:.1f}%"
            }
        
        # Calculate Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(true_probability, decimal_odds)
        
        if kelly_fraction <= 0:
            return {
                "recommended_bet": 0,
                "kelly_fraction": 0,
                "edge": edge * 100,
                "reason": "Negative or zero Kelly fraction"
            }
        
        # Apply conservative scaling
        conservative_fraction = kelly_fraction * self.max_fraction
        
        # Apply confidence scaling
        confidence_adjusted_fraction = conservative_fraction * confidence
        
        # Cap at maximum bet size
        final_fraction = min(confidence_adjusted_fraction, self.max_bet_size)
        
        # Calculate bet amount
        bet_amount = final_fraction * bankroll
        
        # Expected value calculation
        expected_value = (true_probability * (decimal_odds - 1) * bet_amount) - \
                        ((1 - true_probability) * bet_amount)
        
        # Risk metrics
        variance = true_probability * ((decimal_odds - 1) * bet_amount) ** 2 + \
                  (1 - true_probability) * (-bet_amount) ** 2 - expected_value ** 2
        
        standard_deviation = np.sqrt(variance)
        
        return {
            "recommended_bet": round(bet_amount, 2),
            "kelly_fraction": kelly_fraction,
            "conservative_fraction": conservative_fraction,
            "confidence_adjusted_fraction": confidence_adjusted_fraction,
            "final_fraction": final_fraction,
            "edge": edge * 100,
            "expected_value": expected_value,
            "standard_deviation": standard_deviation,
            "risk_reward_ratio": expected_value / standard_deviation if standard_deviation > 0 else 0,
            "bankroll_percentage": final_fraction * 100,
            "decimal_odds": decimal_odds,
            "implied_probability": implied_probability * 100,
            "true_probability": true_probability * 100
        }
    
    def calculate_parlay_kelly(self, individual_bets: list, correlation_matrix: Dict = None) -> Dict:
        """
        Calculate Kelly sizing for correlated parlay bets
        """
        if not individual_bets:
            return {"recommended_bet": 0, "reason": "No bets provided"}
        
        # Calculate independent parlay probability
        independent_prob = 1.0
        for bet in individual_bets:
            independent_prob *= bet["true_probability"]
        
        # Adjust for correlations if provided
        if correlation_matrix:
            # Simplified correlation adjustment
            correlation_adjustment = 1.0
            n_bets = len(individual_bets)
            
            # Average pairwise correlation
            total_correlation = 0
            pairs = 0
            
            for i in range(n_bets):
                for j in range(i + 1, n_bets):
                    correlation = correlation_matrix.get(f"{i}_{j}", 0)
                    total_correlation += correlation
                    pairs += 1
            
            avg_correlation = total_correlation / pairs if pairs > 0 else 0
            
            # Positive correlation increases parlay probability
            correlation_adjustment = 1 + avg_correlation * 0.1  # Max 10% adjustment
            
            adjusted_prob = independent_prob * correlation_adjustment
        else:
            adjusted_prob = independent_prob
        
        # Calculate parlay odds (assuming independent pricing)
        parlay_decimal_odds = 1.0
        for bet in individual_bets:
            individual_decimal_odds = self.american_to_decimal_odds(bet["american_odds"])
            parlay_decimal_odds *= individual_decimal_odds
        
        # Apply Kelly formula to parlay
        kelly_result = self.calculate_optimal_bet_size(
            adjusted_prob, 
            self._decimal_to_american_odds(parlay_decimal_odds),
            individual_bets[0]["bankroll"],  # Assume same bankroll
            min([bet.get("confidence", 1.0) for bet in individual_bets])  # Use lowest confidence
        )
        
        # Add parlay-specific information
        kelly_result.update({
            "parlay_legs": len(individual_bets),
            "independent_probability": independent_prob * 100,
            "correlation_adjusted_probability": adjusted_prob * 100,
            "correlation_effect": (adjusted_prob - independent_prob) * 100,
            "individual_bets": individual_bets
        })
        
        return kelly_result
    
    def _decimal_to_american_odds(self, decimal_odds: float) -> int:
        """Convert decimal odds back to American odds"""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))
    
    def risk_management_check(self, bet_amount: float, bankroll: float, 
                            recent_results: list = None) -> Dict:
        """
        Additional risk management checks
        """
        warnings = []
        
        # Check bet size relative to bankroll
        bet_percentage = (bet_amount / bankroll) * 100
        if bet_percentage > 5:
            warnings.append(f"Bet size {bet_percentage:.1f}% exceeds 5% of bankroll")
        
        # Check recent performance if provided
        if recent_results:
            recent_losses = sum(1 for result in recent_results[-10:] if result < 0)
            if recent_losses >= 7:
                warnings.append("7+ losses in last 10 bets - consider reducing bet size")
        
        # Bankroll drawdown check
        if recent_results and len(recent_results) >= 20:
            peak_bankroll = max(recent_results)
            current_bankroll = recent_results[-1]
            drawdown = (peak_bankroll - current_bankroll) / peak_bankroll
            
            if drawdown > 0.2:  # 20% drawdown
                warnings.append(f"Bankroll down {drawdown*100:.1f}% from peak - consider smaller bets")
        
        return {
            "warnings": warnings,
            "risk_level": "high" if len(warnings) >= 2 else "medium" if warnings else "low",
            "recommended_action": "reduce_bet_size" if warnings else "proceed"
        }


def calculate_bet_size(true_prob: float, american_odds: int, bankroll: float, 
                      confidence: float = 1.0) -> Dict:
    """Quick function to calculate optimal bet size"""
    kelly = KellyCriterion()
    return kelly.calculate_optimal_bet_size(true_prob, american_odds, bankroll, confidence)
