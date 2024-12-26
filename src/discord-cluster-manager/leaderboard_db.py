from typing import Optional

import psycopg2
from consts import (
    DATABASE_URL,
    POSTGRES_DATABASE,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)
from psycopg2 import Error
from utils import LeaderboardItem, SubmissionItem


class LeaderboardDB:
    def __init__(self, host: str, database: str, user: str, password: str, port: str = "5432"):
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
            self.connection = (
                psycopg2.connect(DATABASE_URL, sslmode="require")
                if DATABASE_URL
                else psycopg2.connect(**self.connection_params)
            )
            self.cursor = self.connection.cursor()
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

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

    def create_leaderboard(self, leaderboard: LeaderboardItem) -> Optional[None]:
        try:
            self.cursor.execute(
                """
                INSERT INTO leaderboard.leaderboard (name, deadline, reference_code)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (
                    leaderboard["name"],
                    leaderboard["deadline"],
                    leaderboard["reference_code"],
                ),
            )

            leaderboard_id = self.cursor.fetchone()[0]

            for gpu_type in leaderboard["gpu_types"]:
                self.cursor.execute(
                    """
                    INSERT INTO leaderboard.gpu_type (leaderboard_id, gpu_type)
                    VALUES (%s, %s)
                    """,
                    (leaderboard_id, gpu_type),
                )

            self.connection.commit()
        except psycopg2.Error as e:
            self.connection.rollback()  # Ensure rollback if error occurs
            return f"Error during leaderboard creation: {e}"
        return None

    def delete_leaderboard(self, leaderboard_name: str) -> Optional[str]:
        try:
            # TODO: wait for cascade to be implemented
            self.cursor.execute(
                """
                DELETE FROM leaderboard.leaderboard WHERE name = %s
                """,
                (leaderboard_name,),
            )
            self.connection.commit()
        except psycopg2.Error as e:
            self.connection.rollback()
            return f"Error during leaderboard deletion: {e}"
        return None

    def create_submission(self, submission: SubmissionItem):
        try:
            self.cursor.execute(
                """
                INSERT INTO leaderboard.submission (leaderboard_id, name,
                    user_id, code, submission_time, score)
                VALUES (
                    (SELECT id FROM leaderboard.leaderboard WHERE name = %s),
                    %s, %s, %s, %s, %s)
                """,
                (
                    submission["leaderboard_name"],
                    submission["submission_name"],
                    submission["user_id"],
                    submission["code"],
                    submission["submission_time"],
                    submission["submission_score"],
                ),
            )
            self.connection.commit()
        except psycopg2.Error as e:
            print(f"Error during leaderboard submission: {e}")
            self.connection.rollback()  # Ensure rollback if error occurs

    def get_leaderboards(self) -> list[LeaderboardItem]:
        self.cursor.execute(
            """
            SELECT id, name, deadline, reference_code
            FROM leaderboard.leaderboard
            """
        )

        lbs = self.cursor.fetchall()
        leaderboards = []

        for lb in lbs:
            self.cursor.execute(
                "SELECT * from leaderboard.gpu_type where leaderboard_id = %s", [lb[0]]
            )
            gpu_types = [x[1] for x in self.cursor.fetchall()]

            leaderboards.append(
                LeaderboardItem(
                    id=lb[0], name=lb[1], deadline=lb[2], reference_code=lb[3], gpu_types=gpu_types
                )
            )

        return leaderboards

    def get_leaderboard(self, leaderboard_name: str) -> int | None:
        self.cursor.execute(
            """
            SELECT id, name, deadline, reference_code
            FROM leaderboard.leaderboard
            WHERE name = %s
            """,
            (leaderboard_name,),
        )

        res = self.cursor.fetchone()

        if res:
            return LeaderboardItem(id=res[0], name=res[1], deadline=res[2], reference_code=res[3])
        else:
            return None

    def get_leaderboard_submissions(self, leaderboard_name: str) -> list[SubmissionItem]:
        self.cursor.execute(
            """
            SELECT s.name, s.user_id, s.code, s.submission_time, s.score
            FROM leaderboard.submission s
            JOIN leaderboard.leaderboard l
            ON s.leaderboard_id = l.id
            WHERE l.name = %s
            ORDER BY s.score ASC
            """,
            (leaderboard_name,),
        )

        return [
            SubmissionItem(
                leaderboard_name=leaderboard_name,
                submission_name=submission[0],
                user_id=submission[1],
                code=submission[2],
                submission_time=submission[3],
                submission_score=submission[4],
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
