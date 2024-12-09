import psycopg2
from psycopg2 import Error
from typing import Optional
from utils import LeaderboardItem, SubmissionItem
from consts import (
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DATABASE,
)


class LeaderboardDB:
    def __init__(
        self, host: str, database: str, user: str, password: str, port: str = "5432"
    ):
        """Initialize database connection parameters"""
        self.connection_params = {
            "host": host,
            "database": database,
            "user": user,
            "password": password,
            "port": port,
        }
        self.connection: Optional[psycopg2.extensions.connection] = None
        self.cursor: Optional[psycopg2.extensions.cursor] = None

    def connect(self) -> bool:
        """Establish connection to the database"""
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            self.cursor = self.connection.cursor()
            self._create_tables()
            return True
        except Error as e:
            print(f"Error connecting to PostgreSQL: {e}")
            return False

    def disconnect(self):
        """Close database connection and cursor"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS leaderboard (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            deadline TIMESTAMP NOT NULL,
            reference_code TEXT NOT NULL
        );
        """

        create_submission_table_query = """
        CREATE TABLE IF NOT EXISTS submissions (
            submission_id SERIAL PRIMARY KEY,
            leaderboard_id TEXT NOT NULL,
            submission_name VARCHAR(255) NOT NULL,
            user_id TEXT NOT NULL,
            code TEXT NOT NULL,
            submission_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            submission_score DOUBLE PRECISION NOT NULL,
            FOREIGN KEY (leaderboard_id) REFERENCES leaderboard(name)
        );
        """

        create_run_information_table_query = """
        CREATE TABLE IF NOT EXISTS runinfo (
            submission_id INTEGER NOT NULL,
            stdout TEXT,
            ncu_output TEXT,
            FOREIGN KEY (submission_id) REFERENCES submissions(submission_id)
        );
        """

        try:
            self.cursor.execute(create_table_query)
            self.cursor.execute(create_submission_table_query)
            self.cursor.execute(create_run_information_table_query)
            self.connection.commit()
        except Error as e:
            print(f"Error creating table: {e}")

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

    def create_leaderboard(self, leaderboard: LeaderboardItem):
        try:
            self.cursor.execute(
                """
                INSERT INTO leaderboard (name, deadline, reference_code)
                VALUES (%s, %s, %s)
                """,
                (
                    leaderboard["name"],
                    leaderboard["deadline"],
                    leaderboard["reference_code"],
                ),
            )
            self.connection.commit()
        except psycopg2.Error as e:
            print(f"Error during leaderboard creation: {e}")
            self.connection.rollback()  # Ensure rollback if error occurs

    def create_submission(self, submission: SubmissionItem):
        try:
            self.cursor.execute(
                """
                INSERT INTO submissions (submission_name, submission_time, leaderboard_id, code, user_id, submission_score)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    submission["submission_name"],
                    submission["submission_time"],
                    submission["leaderboard_name"],
                    submission["code"],
                    submission["user_id"],
                    submission["submission_score"],
                ),
            )
            self.connection.commit()
        except psycopg2.Error as e:
            print(f"Error during leaderboard submission: {e}")
            self.connection.rollback()  # Ensure rollback if error occurs

    def get_leaderboards(self) -> list[LeaderboardItem]:
        self.cursor.execute("SELECT * FROM leaderboard")

        return [
            LeaderboardItem(id=lb[0], name=lb[1], deadline=lb[2], reference_code=lb[3])
            for lb in self.cursor.fetchall()
        ]

    def get_leaderboard(self, leaderboard_name: str) -> int | None:
        self.cursor.execute(
            "SELECT * FROM leaderboard WHERE name = %s", (leaderboard_name,)
        )

        res = self.cursor.fetchone()

        if res:
            return LeaderboardItem(
                id=res[0], name=res[1], deadline=res[2], reference_code=res[3]
            )
        else:
            return None

    def get_leaderboard_submissions(
        self, leaderboard_name: str
    ) -> list[SubmissionItem]:
        """
        TODO: Change these all to be IDs instead
        """
        self.cursor.execute(
            "SELECT * FROM submissions WHERE leaderboard_id = %s ORDER BY submission_score ASC",
            (leaderboard_name,),
        )

        return [
            SubmissionItem(
                leaderboard_name=submission[1],
                submission_name=submission[2],
                user_id=submission[3],
                code=submission[4],
                submission_time=submission[5],
                submission_score=submission[6],
            )
            for submission in self.cursor.fetchall()
        ]


if __name__ == "__main__":
    print(
        POSTGRES_HOST,
        POSTGRES_DATABASE,
        POSTGRES_USER,
        POSTGRES_PASSWORD,
        POSTGRES_PORT,
    )

    leaderboard_db = LeaderboardDB(
        POSTGRES_HOST,
        POSTGRES_DATABASE,
        POSTGRES_USER,
        POSTGRES_PASSWORD,
        POSTGRES_PORT,
    )
    leaderboard_db.connect()
    leaderboard_db.disconnect()
