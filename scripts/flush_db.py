#!/usr/bin/env python3

import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
import os

def flush_database():
    # Load environment variables
    load_dotenv()
    
    # Get database connection parameters from environment
    connection_params = {
        "host": os.getenv("POSTGRES_HOST"),
        "database": os.getenv("POSTGRES_DATABASE"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "port": os.getenv("POSTGRES_PORT", "5432")
    }

    # Verify all parameters are present
    missing_params = [k for k, v in connection_params.items() if not v]
    if missing_params:
        print(f"❌ Missing environment variables: {', '.join(missing_params)}")
        return

    try:
        # Connect to database
        print("📡 Connecting to database...")
        connection = psycopg2.connect(**connection_params)
        cursor = connection.cursor()

        # Drop existing tables
        print("🗑️  Dropping existing tables...")
        drop_tables_query = """
        DROP TABLE IF EXISTS submissions CASCADE;
        DROP TABLE IF EXISTS leaderboard CASCADE;
        """
        cursor.execute(drop_tables_query)
        # Commit changes
        connection.commit()
        print("✅ Database flushed and recreated successfully!")

    except Error as e:
        print(f"❌ Database error: {e}")
    finally:
        if 'connection' in locals():
            cursor.close()
            connection.close()
            print("🔌 Database connection closed")

if __name__ == "__main__":
    flush_database() 
