#!/usr/bin/env python3


import psycopg2
from dotenv import load_dotenv
from env import DATABASE_URL, DISABLE_SSL
from psycopg2 import Error


def flush_database():
    # Load environment variables
    load_dotenv()

    if DATABASE_URL is None:
        print("❌ Missing DATABASE_URL environment variable")
        return

    try:
        # Connect to database
        print("📡 Connecting to database...")
        connection = psycopg2.connect(
            DATABASE_URL,
            sslmode="disable" if DISABLE_SSL else "require"
        )
        cursor = connection.cursor()

        # Drop existing tables
        print("🗑️  Dropping existing tables...")
        drop_tables_query = """
        DROP TABLE IF EXISTS submissions CASCADE;
        DROP TABLE IF EXISTS leaderboard CASCADE;
        DROP TABLE IF EXISTS runinfo CASCADE;
        DROP TABLE IF EXISTS _yoyo_log CASCADE;
        DROP TABLE IF EXISTS _yoyo_migration CASCADE;
        DROP TABLE IF EXISTS _yoyo_version CASCADE;
        DROP TABLE IF EXISTS yoyo_lock CASCADE;
        DROP SCHEMA IF EXISTS leaderboard CASCADE;
        """
        cursor.execute(drop_tables_query)
        # Commit changes
        connection.commit()
        print("✅ Database flushed and recreated successfully!")

    except Error as e:
        print(f"❌ Database error: {e}")
    finally:
        if "connection" in locals():
            cursor.close()
            connection.close()
            print("🔌 Database connection closed")


if __name__ == "__main__":
    flush_database()
