"""Main FastAPI application"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from api.routers import predictions, players, slate, weather, health

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting Baseball AI API...")
    yield
    logger.info("Shutting down Baseball AI API...")

# Create FastAPI app
app = FastAPI(
    title="Baseball AI API",
    description="Over/Under prediction system for baseball betting",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(players.router, prefix="/api/players", tags=["players"])
app.include_router(slate.router, prefix="/api/slate", tags=["slate"])
app.include_router(weather.router, prefix="/api/weather", tags=["weather"])

@app.get("/")
async def root():
    return {"message": "Baseball AI API is running"}
