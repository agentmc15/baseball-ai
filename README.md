# Baseball AI - Over/Under Prediction System

A comprehensive baseball analytics system designed specifically for predicting over/under outcomes on Underdog Fantasy.

## Features

- 🔥 Hot/Cold Streak Detection using Hidden Markov Models
- 🌡️ Weather Impact Analysis for accurate predictions
- ⚾ Advanced Pitcher-Batter Matchup Analysis
- 📊 Real-time ML predictions with 55%+ accuracy target
- 💰 Kelly Criterion bankroll management
- 🤖 LangChain/LangGraph agent orchestration

## Repository Structure

```
baseball-ai/
├── backend/          # Python FastAPI backend
├── frontend/         # React TypeScript frontend
├── notebooks/        # Jupyter notebooks for analysis
├── data/            # Data storage (gitignored)
├── docs/            # Documentation
└── tests/           # Test suites
```

## Quick Start

1. **Set up environment**
   ```bash
   cp .env.example .env
   # Add your OpenAI API key and other credentials to .env
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Backend dependencies
   pip install -r backend/requirements/dev.txt
   
   # Frontend dependencies
   cd frontend && npm install
   ```

4. **Run the application**
   ```bash
   # Terminal 1 - Backend
   make run-backend
   
   # Terminal 2 - Frontend
   make run-frontend
   ```

5. **Access the application**
   - Frontend: http://localhost:3000
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Development Commands

- `make help` - Show available commands
- `make test` - Run all tests
- `make format` - Format code
- `make lint` - Run linters
- `make clean` - Clean temporary files

## Architecture

- **Backend**: FastAPI + LangChain/LangGraph
- **Frontend**: React + TypeScript + Tailwind CSS
- **Database**: DuckDB (local, no limits)
- **ML**: XGBoost, Hidden Markov Models
- **Performance**: Rust extensions for critical paths

## License

MIT License
