#!/usr/bin/env python3
"""Initialize the database with required tables"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_tables():
    """Create all required tables"""
    try:
        import duckdb
        from utils.config import settings
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(settings.duckdb_path), exist_ok=True)
        
        # Connect to DuckDB
        conn = duckdb.connect(settings.duckdb_path)
        
        # Create pitches table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pitches (
                pitch_id BIGINT PRIMARY KEY,
                game_id VARCHAR(20) NOT NULL,
                at_bat_id INTEGER NOT NULL,
                pitch_number INTEGER NOT NULL,
                pitcher_id INTEGER NOT NULL,
                batter_id INTEGER NOT NULL,
                pitch_type VARCHAR(5),
                velocity DECIMAL(4,1),
                spin_rate INTEGER,
                plate_x DECIMAL(4,2),
                plate_z DECIMAL(4,2),
                balls INTEGER,
                strikes INTEGER,
                game_date DATE NOT NULL
            )
        """)
        
        # Create player performance table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS player_performance_daily (
                player_id INTEGER,
                date DATE,
                plate_appearances INTEGER,
                hits INTEGER,
                total_bases INTEGER,
                runs INTEGER,
                rbis INTEGER,
                strikeouts INTEGER,
                PRIMARY KEY (player_id, date)
            )
        """)
        
        # Create predictions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id BIGINT PRIMARY KEY,
                player_id INTEGER,
                game_date DATE,
                stat_type VARCHAR(20),
                line DECIMAL(5,2),
                prediction DECIMAL(5,2),
                confidence DECIMAL(3,2),
                edge DECIMAL(4,3),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Close connection
        conn.close()
        
        print("✓ Database tables created successfully")
        
    except ImportError:
        print("⚠ Database setup skipped - install dependencies first")
        print("Run: pip install -r backend/requirements/base.txt")

def main():
    """Main setup function"""
    print("Setting up database...")
    create_tables()

if __name__ == "__main__":
    main()
