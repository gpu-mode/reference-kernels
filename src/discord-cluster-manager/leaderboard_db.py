import json
import logging
from typing import List, Optional

import discord
import psycopg2
from env import (
    DATABASE_URL,
    POSTGRES_DATABASE,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)
from psycopg2 import Error
from task import LeaderboardTask, build_from_legacy_reference
from utils import LeaderboardItem, LRUCache, SubmissionItem

leaderboard_name_cache = LRUCache(max_size=512)


async def leaderboard_name_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[discord.app_commands.Choice[str]]:
    """Return leaderboard names that match the current typed name"""
    try:
        cached_value = leaderboard_name_cache[current]
        if cached_value is not None:
            return cached_value

        bot = interaction.client
        with bot.leaderboard_db as db:
            leaderboards = db.get_leaderboard_names()
        filtered = [lb for lb in leaderboards if current.lower() in lb.lower()]
        leaderboard_name_cache[current] = [
            discord.app_commands.Choice(name=name, value=name) for name in filtered[:25]
        ]
        return leaderboard_name_cache[current]
    except Exception as e:
        logging.exception("Error in leaderboard autocomplete", exc_info=e)
        return []


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
                psycopg2.connect(DATABASE_URL)
                if DATABASE_URL
                else psycopg2.connect(**self.connection_params, sslmode="require")
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
                INSERT INTO leaderboard.leaderboard (name, deadline, task, creator_id)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (
                    leaderboard["name"],
                    leaderboard["deadline"],
                    leaderboard["task"].to_str(),
                    leaderboard["creator_id"],
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
            leaderboard_name_cache.invalidate()  # Invalidate autocomplete cache
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
            leaderboard_name_cache.invalidate()  # Invalidate autocomplete cache
        except psycopg2.Error as e:
            self.connection.rollback()
            return f"Error during leaderboard deletion: {e}"
        return None

    def create_submission(self, submission: SubmissionItem):
        try:
            self.cursor.execute(
                """
                INSERT INTO leaderboard.submission (leaderboard_id, name,
                    user_id, code, submission_time, score, gpu_type, stdout,
                    profiler_output)
                VALUES (
                    (SELECT id FROM leaderboard.leaderboard WHERE name = %s),
                    %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    submission["leaderboard_name"],
                    submission["submission_name"],
                    submission["user_id"],
                    submission["code"],
                    submission["submission_time"],
                    submission["submission_score"],
                    submission["gpu_type"],
                    submission.get("stdout", None),
                    submission.get("profiler_output", None),
                ),
            )
            self.connection.commit()
        except psycopg2.Error as e:
            print(f"Error during leaderboard submission: {e}")
            self.connection.rollback()  # Ensure rollback if error occurs

    def get_leaderboard_names(self) -> list[str]:
        self.cursor.execute("SELECT name FROM leaderboard.leaderboard")
        return [x[0] for x in self.cursor.fetchall()]

    def get_leaderboards(self) -> list[LeaderboardItem]:
        self.cursor.execute(
            """
            SELECT id, name, deadline, task, creator_id
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
                    id=lb[0],
                    name=lb[1],
                    deadline=lb[2],
                    task=LeaderboardTask.from_dict(lb[3]),
                    gpu_types=gpu_types,
                    creator_id=lb[4],
                )
            )

        return leaderboards

    def get_leaderboard_gpu_types(self, leaderboard_name: str) -> List[str] | None:
        self.cursor.execute(
            """
            SELECT *
            FROM leaderboard.gpu_type
            WHERE leaderboard_id = (
                SELECT id
                FROM leaderboard.leaderboard
                WHERE name = %s
            )
            """,
            (leaderboard_name,),
        )

        gpu_types = [x[1] for x in self.cursor.fetchall()]

        if gpu_types:
            return gpu_types
        else:
            return None

    def get_leaderboard(self, leaderboard_name: str) -> LeaderboardItem | None:
        self.cursor.execute(
            """
            SELECT id, name, deadline, task, creator_id
            FROM leaderboard.leaderboard
            WHERE name = %s
            """,
            (leaderboard_name,),
        )

        res = self.cursor.fetchone()

        if res:
            # TODO: This is just a clutch to keep compatibility with old leaderboards
            try:
                task = LeaderboardTask.from_dict(res[3])
            except json.JSONDecodeError:
                logging.error("json decoding error in LB %s. Legacy task?", leaderboard_name)
                task = build_from_legacy_reference(res[3])

            return LeaderboardItem(
                id=res[0],
                name=res[1],
                deadline=res[2],
                task=task,
                creator_id=res[4],
            )
        else:
            return None

    def get_leaderboard_submissions(
        self, leaderboard_name: str, gpu_name: str, user_id: Optional[str] = None
    ) -> list[SubmissionItem]:
        query = """
            WITH ranked_submissions AS (
                SELECT
                    s.name,
                    s.user_id,
                    s.code,
                    s.submission_time,
                    s.score,
                    s.gpu_type,
                    RANK() OVER (ORDER BY s.score ASC) as rank
                FROM leaderboard.submission s
                JOIN leaderboard.leaderboard l ON s.leaderboard_id = l.id
                WHERE l.name = %s AND s.gpu_type = %s
            )
            SELECT * FROM ranked_submissions
            """
        if user_id:
            query += " WHERE user_id = %s"

        query += " ORDER BY score ASC"

        args = (leaderboard_name, gpu_name)
        if user_id:
            args = args + (user_id,)

        self.cursor.execute(query, args)

        return [
            SubmissionItem(
                leaderboard_name=leaderboard_name,
                submission_name=submission[0],
                user_id=submission[1],
                code=submission[2],
                submission_time=submission[3],
                submission_score=submission[4],
                gpu_type=gpu_name,
                rank=submission[6],
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
