.PHONY: help setup run test clean format lint

help:
	@echo "Available commands:"
	@echo "  make setup    - Set up development environment"
	@echo "  make run      - Run the application with Docker"
	@echo "  make run-dev  - Run in development mode"
	@echo "  make test     - Run all tests"
	@echo "  make train    - Train ML models"
	@echo "  make backtest - Run backtesting"
	@echo "  make clean    - Clean temporary files"
	@echo "  make format   - Format code"
	@echo "  make lint     - Run linters"

setup:
	@echo "Setting up development environment..."
	cp -n .env.example .env || true
	python3 -m venv venv
	source venv/bin/activate && pip install -r backend/requirements/dev.txt
	cd frontend && npm install

run:
	docker-compose up

run-dev:
	docker-compose -f docker-compose.dev.yml up

run-backend:
	cd backend && uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	cd frontend && npm start

test:
	cd backend && pytest
	cd frontend && npm test

test-backend:
	cd backend && pytest -v --cov=. --cov-report=html

train:
	python backend/scripts/train_models.py

backtest:
	python backend/models/training/backtesting.py --days 30

update-data:
	python backend/scripts/daily_update.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf data/cache/* 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	rm -rf htmlcov 2>/dev/null || true

format:
	cd backend && black . && isort .
	cd frontend && npm run format

lint:
	cd backend && flake8 . && mypy .
	cd frontend && npm run lint
