"""
Betting strategy modules
"""
from .kelly_criterion import KellyCriterion, calculate_bet_size

__all__ = [
    "KellyCriterion",
    "calculate_bet_size"
]
