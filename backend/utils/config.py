"""Configuration management"""
import os
from typing import Optional

class Settings:
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openweather_api_key: str = os.getenv("OPENWEATHER_API_KEY", "")
    
    # Database
    duckdb_path: str = os.getenv("DUCKDB_PATH", "./data/baseball.db")
    
    # Redis
    redis_url: Optional[str] = os.getenv("REDIS_URL")
    
    # API Settings
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    
    # Data Sources
    mlb_api_delay: float = float(os.getenv("MLB_API_DELAY", "1.0"))
    pybaseball_cache: str = os.getenv("PYBASEBALL_CACHE", "./data/cache/pybaseball")
    
    # Model Settings
    model_update_schedule: str = os.getenv("MODEL_UPDATE_SCHEDULE", "0 6 * * *")
    backtest_days: int = int(os.getenv("BACKTEST_DAYS", "30"))
    
    # Betting Settings
    min_edge_threshold: float = float(os.getenv("MIN_EDGE_THRESHOLD", "0.03"))
    kelly_fraction: float = float(os.getenv("KELLY_FRACTION", "0.25"))
    
    # LangChain
    langchain_tracing_v2: bool = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true"
    langchain_endpoint: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    langchain_api_key: Optional[str] = os.getenv("LANGCHAIN_API_KEY")
    langchain_project: str = os.getenv("LANGCHAIN_PROJECT", "baseball-ai")

settings = Settings()
