"""
MLB Stats API Data Source
"""
import requests
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import time

from utils.config import settings


class MLBDataSource:
    """Interface to MLB Stats API"""
    
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.delay = settings.mlb_api_delay
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make rate-limited request to MLB API"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            time.sleep(self.delay)  # Rate limiting
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
            return {}
    
    def get_todays_games(self) -> List[Dict]:
        """Get today's MLB games"""
        today = date.today().strftime("%Y-%m-%d")
        data = self._make_request("schedule", {"date": today})
        
        games = []
        for game_date in data.get("dates", []):
            for game in game_date.get("games", []):
                games.append({
                    "game_id": game["gamePk"],
                    "home_team": game["teams"]["home"]["team"]["name"],
                    "away_team": game["teams"]["away"]["team"]["name"],
                    "game_time": game["gameDate"],
                    "status": game["status"]["detailedState"],
                    "venue_id": game["venue"]["id"]
                })
        
        return games
    
    def get_player_recent_stats(self, player_id: int, days: int = 15) -> List[Dict]:
        """Get player's recent game logs"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        params = {
            "personId": player_id,
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
            "gameType": "R"  # Regular season only
        }
        
        data = self._make_request(f"people/{player_id}/stats", params)
        
        # Parse game logs
        games = []
        stats = data.get("stats", [])
        if stats:
            for split in stats[0].get("splits", []):
                stat = split.get("stat", {})
                games.append({
                    "date": split.get("date"),
                    "hits": stat.get("hits", 0),
                    "total_bases": stat.get("totalBases", 0),
                    "runs": stat.get("runs", 0),
                    "rbis": stat.get("rbi", 0),
                    "strikeouts": stat.get("strikeOuts", 0),
                    "at_bats": stat.get("atBats", 0)
                })
        
        return games
    
    def get_batter_pitcher_matchup(self, batter_id: int, pitcher_id: int) -> Dict:
        """Get historical matchup data"""
        # MLB API doesn't directly provide H2H, so we'll structure for future implementation
        return {
            "batter_id": batter_id,
            "pitcher_id": pitcher_id,
            "at_bats": 0,  # Would be calculated from historical data
            "stats": {},
            "pitcher_hand": "R",  # Would get from player data
            "batter_vs_hand_stats": {}
        }
    
    def get_game_weather(self, game_id: str) -> Dict:
        """Get weather data for a specific game"""
        # Weather data would come from external API
        # This is a placeholder structure
        return {
            "temperature": 72,
            "wind_speed": 8,
            "wind_direction": "out_to_left",
            "humidity": 65,
            "pressure": 30.15,
            "conditions": "clear"
        }
    
    def get_team_roster(self, team_id: int) -> List[Dict]:
        """Get current team roster"""
        data = self._make_request(f"teams/{team_id}/roster")
        
        players = []
        for person in data.get("roster", []):
            player = person.get("person", {})
            players.append({
                "player_id": player.get("id"),
                "name": player.get("fullName"),
                "position": person.get("position", {}).get("name"),
                "jersey_number": person.get("jerseyNumber")
            })
        
        return players
    
    def get_starting_lineups(self, game_id: str) -> Dict:
        """Get starting lineups for a game"""
        data = self._make_request(f"game/{game_id}/boxscore")
        
        lineups = {"home": [], "away": []}
        
        teams = data.get("teams", {})
        for team_type in ["home", "away"]:
            team_data = teams.get(team_type, {})
            batters = team_data.get("batters", [])
            
            for batter_id in batters[:9]:  # Starting 9
                player_data = team_data.get("players", {}).get(f"ID{batter_id}", {})
                lineups[team_type].append({
                    "player_id": batter_id,
                    "name": player_data.get("person", {}).get("fullName"),
                    "batting_order": len(lineups[team_type]) + 1
                })
        
        return lineups
