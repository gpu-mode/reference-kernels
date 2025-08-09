import dataclasses
import datetime
import json
from typing import Dict, List, Optional

import psycopg2

from libkernelbot.db_types import LeaderboardItem, LeaderboardRankedEntry, RunItem, SubmissionItem
from libkernelbot.run_eval import CompileResult, RunResult, SystemInfo
from libkernelbot.task import LeaderboardDefinition, LeaderboardTask
from libkernelbot.utils import (
    KernelBotError,
    LRUCache,
    setup_logging,
)

logger = setup_logging(__name__)


class LeaderboardDB:
    def __init__(
        self, host: str, database: str, user: str, password: str, port: str, url: str, ssl_mode: str
    ):
        """Initialize database connection parameters"""
        self.connection_params = {
            "host": host,
            "database": database,
            "user": user,
            "password": password,
            "port": port,
        }
        self.url = url
        self.ssl_mode = ssl_mode
        self.connection: Optional[psycopg2.extensions.connection] = None
        self.refcount: int = 0
        self.cursor: Optional[psycopg2.extensions.cursor] = None
        self.name_cache = LRUCache(max_size=512)

    def connect(self) -> bool:
        """Establish connection to the database"""
        try:
            self.connection = (
                psycopg2.connect(self.url, sslmode=self.ssl_mode)
                if self.url
                else psycopg2.connect(**self.connection_params)
            )
            self.cursor = self.connection.cursor()
            return True
        except psycopg2.Error as e:
            logger.exception("Error connecting to PostgreSQL", exc_info=e)
            return False

    def disconnect(self):
        """Close database connection and cursor"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        self.cursor = None
        self.connection = None

    def __enter__(self) -> "LeaderboardDB":
        """Context manager entry"""
        if self.connection is not None:
            self.refcount += 1
            return self

        if self.connect():
            self.refcount = 1
            return self

        raise KernelBotError("Could not connect to database", code=500)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.refcount -= 1
        if self.refcount == 0:
            self.disconnect()

    def create_leaderboard(
        self,
        *,
        name: str,
        deadline: datetime.datetime,
        definition: LeaderboardDefinition,
        creator_id: int,
        forum_id: int,
        gpu_types: list | str,
    ) -> int:
        try:
            task = definition.task
            self.cursor.execute(
                """
                INSERT INTO leaderboard.leaderboard (name, deadline, task, creator_id,
                                                     forum_id, description)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (name, deadline, task.to_str(), creator_id, forum_id, definition.description),
            )

            leaderboard_id = self.cursor.fetchone()[0]

            if isinstance(gpu_types, str):
                gpu_types = [gpu_types]

            for gpu_type in gpu_types:
                self.cursor.execute(
                    """
                    INSERT INTO leaderboard.gpu_type (leaderboard_id, gpu_type)
                    VALUES (%s, %s)
                    """,
                    (leaderboard_id, gpu_type),
                )

            # insert templates
            for lang, code in definition.templates.items():
                self.cursor.execute(
                    """
                    INSERT INTO leaderboard.templates (leaderboard_id, lang, code)
                    VALUES (%s, %s, %s)
                    """,
                    (leaderboard_id, lang, code),
                )

            self.connection.commit()
            self.name_cache.invalidate()  # Invalidate autocomplete cache
            return leaderboard_id
        except psycopg2.Error as e:
            logger.exception("Error in leaderboard creation.", e)
            if isinstance(e, psycopg2.errors.UniqueViolation):
                raise KernelBotError(
                    "Error: Tried to create a leaderboard " f'"{name}" that already exists.'
                ) from e
            self.connection.rollback()  # Ensure rollback if error occurs
            raise KernelBotError("Error in leaderboard creation.") from e

    def update_leaderboard(
        self, name, deadline: datetime.datetime, definition: LeaderboardDefinition
    ):
        task = definition.task
        try:
            self.cursor.execute(
                """
                UPDATE leaderboard.leaderboard
                SET deadline = %s, task = %s, description = %s
                WHERE name = %s;
                """,
                (
                    deadline.astimezone(datetime.timezone.utc),
                    task.to_str(),
                    definition.description,
                    name,
                ),
            )

            self.cursor.execute(
                """
                SELECT id
                FROM leaderboard.leaderboard
                WHERE name = %s
                """,
                (name,),
            )

            lb_id = self.cursor.fetchone()[0]

            # replace templates
            self.cursor.execute(
                """
                DELETE FROM leaderboard.templates
                WHERE leaderboard_id = %s
                """,
                (lb_id,),
            )

            for lang, code in definition.templates.items():
                self.cursor.execute(
                    """
                    INSERT INTO leaderboard.templates (leaderboard_id, lang, code)
                    VALUES (%s, %s, %s)
                    """,
                    (lb_id, lang, code),
                )

            self.connection.commit()
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.exception("Error during leaderboard update", exc_info=e)
            raise KernelBotError("Error during leaderboard update") from e

    def delete_leaderboard(self, leaderboard_name: str, force: bool = False):
        try:
            if force:
                self.cursor.execute(
                    """
                    DELETE FROM leaderboard.runs
                    WHERE submission_id IN (
                        SELECT leaderboard.submission.id
                        FROM leaderboard.submission
                        WHERE leaderboard.submission.leaderboard_id IN (
                            SELECT leaderboard.leaderboard.id FROM leaderboard.leaderboard
                                WHERE leaderboard.leaderboard.name = %s
                        )
                    );
""",
                    (leaderboard_name,),
                )
                self.cursor.execute(
                    """
                    DELETE FROM leaderboard.submission
                    USING leaderboard.leaderboard
                    WHERE leaderboard.submission.leaderboard_id = leaderboard.leaderboard.id
                        AND leaderboard.leaderboard.name = %s;
                    """,
                    (leaderboard_name,),
                )

            self.cursor.execute(
                """
                DELETE FROM leaderboard.templates
                USING leaderboard.leaderboard
                WHERE leaderboard.templates.leaderboard_id = leaderboard.leaderboard.id
                    AND leaderboard.leaderboard.name = %s;
                """,
                (leaderboard_name,),
            )
            self.cursor.execute(
                """
                DELETE FROM leaderboard.leaderboard WHERE name = %s
                """,
                (leaderboard_name,),
            )
            self.connection.commit()
            self.name_cache.invalidate()  # Invalidate autocomplete cache
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.exception("Could not delete leaderboard %s.", leaderboard_name, exc_info=e)
            raise KernelBotError(f"Could not delete leaderboard {leaderboard_name}.") from e

    def create_submission(
        self,
        leaderboard: str,
        file_name: str,
        user_id: int,
        code: str,
        time: datetime.datetime,
        user_name: str = None,
    ) -> Optional[int]:
        try:
            # check if we already have the code
            self.cursor.execute(
                """
                SELECT id, code
                FROM leaderboard.code_files
                WHERE hash = encode(sha256(%s), 'hex')
                """,
                (code.encode("utf-8"),),
            )

            code_id = None
            for candidate in self.cursor.fetchall():
                if candidate[1] == code:
                    code_id = candidate[0]
                    break

            if code_id is None:
                # a genuinely new submission
                self.cursor.execute(
                    """
                    INSERT INTO leaderboard.code_files (CODE)
                    VALUES (%s)
                    RETURNING id
                    """,
                    (code.encode("utf-8"),),
                )
                code_id = self.cursor.fetchone()
            # Check if user exists in user_info, if not add them
            self.cursor.execute(
                """
                SELECT 1 FROM leaderboard.user_info WHERE id = %s
                """,
                (str(user_id),),
            )
            if not self.cursor.fetchone():
                self.cursor.execute(
                    """
                    INSERT INTO leaderboard.user_info (id, user_name)
                    VALUES (%s, %s)
                    """,
                    (str(user_id), user_name),
                )
            self.cursor.execute(
                """
                INSERT INTO leaderboard.submission (leaderboard_id, file_name,
                    user_id, code_id, submission_time)
                VALUES (
                    (SELECT id FROM leaderboard.leaderboard WHERE name = %s),
                    %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    leaderboard,
                    file_name,
                    user_id,
                    code_id,
                    time,
                ),
            )
            submission_id = self.cursor.fetchone()[0]
            assert submission_id is not None
            self.connection.commit()
            return submission_id
        except psycopg2.Error as e:
            logger.error(
                "Error during creation of submission for leaderboard '%s' by user '%s'",
                leaderboard,
                user_id,
                exc_info=e,
            )
            self.connection.rollback()  # Ensure rollback if error occurs
            raise KernelBotError("Error during creation of submission") from e

    def mark_submission_done(
        self,
        submission: int,
    ) -> Optional[int]:
        try:
            self.cursor.execute(
                """
                UPDATE leaderboard.submission
                SET done = TRUE
                WHERE id = %s
                """,
                (submission,),
            )
            self.connection.commit()
        except psycopg2.Error as e:
            logger.error("Could not mark submission '%s' as done.", submission, exc_info=e)
            self.connection.rollback()  # Ensure rollback if error occurs
            raise KernelBotError("Error while finalizing submission") from e

    def create_submission_run(
        self,
        submission: int,
        start: datetime.datetime,
        end: datetime.datetime,
        mode: str,
        secret: bool,
        runner: str,
        score: Optional[float],
        compilation: Optional[CompileResult],
        result: RunResult,
        system: SystemInfo,
    ):
        try:
            if compilation is not None:
                compilation = json.dumps(dataclasses.asdict(compilation))

            meta = {
                k: result.__dict__[k]
                for k in ["stdout", "stderr", "success", "exit_code", "command", "duration"]
            }
            self.cursor.execute(
                """
                INSERT INTO leaderboard.runs (submission_id, start_time, end_time, mode,
                secret, runner, score, passed, compilation, meta, result, system_info
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    submission,
                    start,
                    end,
                    mode,
                    secret,
                    runner,
                    score,
                    result.passed,
                    compilation,
                    json.dumps(meta),
                    json.dumps(result.result),
                    json.dumps(dataclasses.asdict(system)),
                ),
            )
            self.connection.commit()
        except psycopg2.Error as e:
            logger.exception(
                "Error during adding %s run on %s for submission '%s'",
                mode,
                runner,
                submission,
                exc_info=e,
            )
            self.connection.rollback()  # Ensure rollback if error occurs
            raise KernelBotError("Could not create leaderboard submission entry in database") from e

    def get_leaderboard_names(self, active_only: bool = False) -> list[str]:
        if active_only:
            self.cursor.execute(
                "SELECT name FROM leaderboard.leaderboard WHERE leaderboard.deadline > %s",
                (datetime.datetime.now().astimezone(datetime.timezone.utc),),
            )
        else:
            self.cursor.execute("SELECT name FROM leaderboard.leaderboard")
        return [x[0] for x in self.cursor.fetchall()]

    def get_leaderboards(self) -> list["LeaderboardItem"]:
        self.cursor.execute(
            """
            SELECT id, name, deadline, task, creator_id, forum_id, description
            FROM leaderboard.leaderboard
            """
        )

        lbs = self.cursor.fetchall()
        leaderboards = []

        for lb in lbs:
            self.cursor.execute(
                "SELECT * from leaderboard.gpu_type WHERE leaderboard_id = %s", [lb[0]]
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
                    forum_id=lb[5],
                    description=lb[6],
                )
            )

        return leaderboards

    def get_leaderboard_gpu_types(self, leaderboard_name: str) -> List[str]:
        self.cursor.execute(
            """
            SELECT id
            FROM leaderboard.leaderboard
            WHERE name = %s
            """,
            (leaderboard_name,),
        )
        lb_id = self.cursor.fetchone()
        if lb_id is None:
            raise LeaderboardDoesNotExist(leaderboard_name)

        self.cursor.execute(
            """
            SELECT gpu_type
            FROM leaderboard.gpu_type
            WHERE leaderboard_id = %s
            """,
            (lb_id[0],),
        )

        return [x[0] for x in self.cursor.fetchall()]

    def get_leaderboard_templates(self, leaderboard_name: str) -> Dict[str, str]:
        self.cursor.execute(
            """
            SELECT id
            FROM leaderboard.leaderboard
            WHERE name = %s
            """,
            (leaderboard_name,),
        )
        lb_id = self.cursor.fetchone()
        if lb_id is None:
            raise LeaderboardDoesNotExist(leaderboard_name)

        self.cursor.execute(
            """
            SELECT lang, code
            FROM leaderboard.templates
            WHERE leaderboard_id = %s
            """,
            (lb_id[0],),
        )

        return {x[0]: x[1] for x in self.cursor.fetchall()}

    def get_leaderboard(self, leaderboard_name: str) -> "LeaderboardItem":
        self.cursor.execute(
            """
            SELECT id, name, deadline, task, creator_id, forum_id, secret_seed, description
            FROM leaderboard.leaderboard
            WHERE name = %s
            """,
            (leaderboard_name,),
        )

        res = self.cursor.fetchone()

        if res:
            task = LeaderboardTask.from_dict(res[3])
            return LeaderboardItem(
                id=res[0],
                name=res[1],
                deadline=res[2],
                task=task,
                creator_id=res[4],
                forum_id=res[5],
                secret_seed=res[6],
                gpu_types=self.get_leaderboard_gpu_types(res[1]),
                description=res[7],
            )
        else:
            raise LeaderboardDoesNotExist(leaderboard_name)

    def get_leaderboard_submissions(
        self,
        leaderboard_name: str,
        gpu_name: str,
        user_id: Optional[str] = None,
        limit: int = None,
        offset: int = 0,
    ) -> list["LeaderboardRankedEntry"]:
        # separate cases, for personal we want all submissions, for general we want best per user
        if user_id:
            # Query all if user_id (means called from show-personal)
            query = """
                SELECT
                    s.file_name,
                    s.id,
                    s.user_id,
                    s.submission_time,
                    r.score,
                    r.runner,
                    ui.user_name,
                    RANK() OVER (ORDER BY r.score ASC) as rank
                FROM leaderboard.runs r
                JOIN leaderboard.submission s ON r.submission_id = s.id
                JOIN leaderboard.leaderboard l ON s.leaderboard_id = l.id
                JOIN leaderboard.user_info ui ON s.user_id = ui.id
                WHERE l.name = %s
                    AND r.runner = %s
                    AND NOT r.secret
                    AND r.score IS NOT NULL
                    AND r.passed
                    AND s.user_id = %s
                ORDER BY r.score ASC
                LIMIT %s OFFSET %s
                """
            args = (leaderboard_name, gpu_name, user_id, limit, offset)
        else:
            # Query best submission per user if no user_id (means called from show)
            query = """
                WITH best_submissions AS (
                    SELECT DISTINCT ON (s.user_id)
                        s.id as submission_id,
                        s.file_name,
                        s.user_id,
                        s.submission_time,
                        r.score,
                        r.runner
                    FROM leaderboard.runs r
                    JOIN leaderboard.submission s ON r.submission_id = s.id
                    JOIN leaderboard.leaderboard l ON s.leaderboard_id = l.id
                    JOIN leaderboard.user_info ui ON s.user_id = ui.id
                    WHERE l.name = %s AND r.runner = %s AND NOT r.secret
                          AND r.score IS NOT NULL AND r.passed
                    ORDER BY s.user_id, r.score ASC
                )
                SELECT
                    bs.file_name,
                    bs.submission_id,
                    bs.user_id,
                    bs.submission_time,
                    bs.score,
                    bs.runner,
                    ui.user_name,
                    RANK() OVER (ORDER BY bs.score ASC) as rank
                FROM best_submissions bs
                JOIN leaderboard.user_info ui ON bs.user_id = ui.id
                ORDER BY bs.score ASC
                LIMIT %s OFFSET %s
                """
            args = (leaderboard_name, gpu_name, limit, offset)

        self.cursor.execute(query, args)

        return [
            LeaderboardRankedEntry(
                submission_name=submission[0],
                submission_id=submission[1],
                user_id=submission[2],
                submission_time=submission[3],
                submission_score=submission[4],
                user_name=submission[6],
                rank=submission[7],
                leaderboard_name=leaderboard_name,
                gpu_type=gpu_name,
            )
            for submission in self.cursor.fetchall()
        ]

    def generate_stats(self, last_day: bool):
        try:
            return self._generate_stats(last_day)
        except Exception as e:
            logger.exception("error generating stats", exc_info=e)
            raise

    def _generate_runner_stats(self, last_day: bool = False):
        select_expr = "WHERE NOW() - s.submission_time <= interval '24 hours'" if last_day else ""
        # per-runner stats
        self.cursor.execute(
            f"""
            SELECT
                runner,
                COUNT(*),
                COUNT(*) FILTER (WHERE passed),
                COUNT(score),
                COUNT(*) FILTER (WHERE secret),
                MAX(runs.start_time - s.submission_time),
                AVG(runs.start_time - s.submission_time),
                SUM(runs.end_time - runs.start_time)
            FROM leaderboard.runs JOIN leaderboard.submission s ON submission_id = s.id
            {select_expr}
            GROUP BY runner;
            """
        )

        result = {}
        for row in self.cursor.fetchall():
            result[f"num_run.{row[0]}"] = row[1]
            result[f"runs_passed.{row[0]}"] = row[2]
            result[f"runs_scored.{row[0]}"] = row[3]
            result[f"runs_secret.{row[0]}"] = row[4]
            result[f"max_delay.{row[0]}"] = row[5]
            result[f"avg_delay.{row[0]}"] = row[6]
            result[f"total_runtime.{row[0]}"] = row[7]

        return result

    def _generate_submission_stats(self, last_day: bool = False):
        select_expr = "WHERE NOW() - submission_time <= interval '24 hours'" if last_day else ""
        self.cursor.execute(
            f"""
            SELECT
                COUNT(*),
                COUNT(*) FILTER (WHERE NOT done),
                COUNT(DISTINCT user_id)
            FROM leaderboard.submission
            {select_expr}
            ;
            """
        )
        num_sub, num_sub_wait, num_users = self.cursor.fetchone()
        return {
            "num_submissions": num_sub,
            "sub_waiting": num_sub_wait,
            "num_users": num_users,
        }

    def _generate_stats(self, last_day: bool = False):
        result = self._generate_submission_stats(last_day)
        result.update(self._generate_runner_stats(last_day))

        # code-level stats
        if not last_day:
            self.cursor.execute(
                """
                SELECT COUNT(*) FROM leaderboard.code_files;
                """
            )
            result["num_unique_codes"] = self.cursor.fetchone()[0]

        else:
            # calculate heavy hitters
            self.cursor.execute(
                """
                WITH run_durations AS (
                    SELECT
                        s.user_id AS user_id,
                        r.end_time - r.start_time AS duration
                    FROM leaderboard.runs r
                    JOIN leaderboard.submission s ON r.submission_id = s.id
                    WHERE NOW() - s.submission_time <= interval '24 hours'
                )
                SELECT
                    user_id,
                    SUM(duration) AS total
                FROM run_durations
                GROUP BY user_id
                ORDER BY total DESC
                LIMIT 10;
                """
            )

            for row in self.cursor.fetchall():
                result[f"total.{row[0]}"] = row[1]

        return result

    def get_user_from_id(self, id: str) -> Optional[str]:
        try:
            self.cursor.execute(
                """
                SELECT user_name FROM leaderboard.user_info WHERE id = %s
                """,
                (id,),
            )
            return self.cursor.fetchone()[0]
        except Exception:
            return None

    def delete_submission(self, submission_id: int):
        try:
            # first, the runs
            query = """
                    DELETE FROM leaderboard.runs
                    WHERE submission_id = %s
                    """
            self.cursor.execute(query, (submission_id,))

            # next, the submission itself
            query = """
                   DELETE FROM leaderboard.submission
                   WHERE id = %s
                   """
            self.cursor.execute(query, (submission_id,))

            # TODO delete code file? Could be one-to-many mapping, so we'd need
            # to figure out if it is used elsewhere first.
            self.connection.commit()
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.exception("Could not delete submission %s.", submission_id, exc_info=e)
            raise KernelBotError(f"Could not delete submission {submission_id}!") from e

    def get_submission_by_id(self, submission_id: int) -> Optional["SubmissionItem"]:
        query = """
                SELECT s.leaderboard_id, lb.name, s.file_name, s.user_id,
                       s.submission_time, s.done, c.code
                FROM leaderboard.submission s
                JOIN leaderboard.code_files c ON s.code_id = c.id
                JOIN leaderboard.leaderboard lb ON s.leaderboard_id = lb.id
                WHERE s.id = %s
                """
        self.cursor.execute(query, (submission_id,))
        submission = self.cursor.fetchone()
        if submission is None:
            return None

        # OK, now get the runs
        query = """
                SELECT start_time, end_time, mode, secret, runner, score,
                       passed, compilation, meta, result, system_info
                FROM leaderboard.runs
                WHERE submission_id = %s
                """
        self.cursor.execute(query, (submission_id,))
        runs = self.cursor.fetchall()

        runs = [
            RunItem(
                start_time=r[0],
                end_time=r[1],
                mode=r[2],
                secret=r[3],
                runner=r[4],
                score=r[5],
                passed=r[6],
                compilation=r[7],
                meta=r[8],
                result=r[9],
                system=r[10],
            )
            for r in runs
        ]

        return SubmissionItem(
            submission_id=submission_id,
            leaderboard_id=submission[0],
            leaderboard_name=submission[1],
            file_name=submission[2],
            user_id=submission[3],
            submission_time=submission[4],
            done=submission[5],
            code=submission[6],
            runs=runs,
        )

    def get_leaderboard_submission_count(
        self,
        leaderboard_name: str,
        gpu_name: str,
        user_id: Optional[str] = None,
    ) -> int:
        """Get the total count of submissions for a leaderboard"""
        if user_id:
            query = """
                SELECT COUNT(*)
                FROM leaderboard.runs r
                JOIN leaderboard.submission s ON r.submission_id = s.id
                JOIN leaderboard.leaderboard l ON s.leaderboard_id = l.id
                WHERE l.name = %s
                    AND r.runner = %s
                    AND NOT r.secret
                    AND r.score IS NOT NULL
                    AND r.passed
                    AND s.user_id = %s
                """
            args = (leaderboard_name, gpu_name, user_id)
        else:
            query = """
                SELECT COUNT(DISTINCT s.user_id)
                FROM leaderboard.runs r
                JOIN leaderboard.submission s ON r.submission_id = s.id
                JOIN leaderboard.leaderboard l ON s.leaderboard_id = l.id
                WHERE l.name = %s
                    AND r.runner = %s
                    AND NOT r.secret
                    AND r.score IS NOT NULL
                    AND r.passed
                """
            args = (leaderboard_name, gpu_name)

        self.cursor.execute(query, args)
        return self.cursor.fetchone()[0]

    def init_user_from_cli(self, cli_id: str, auth_provider: str):
        """
        Initialize a user from CLI authentication flow.
        Checks if cli_id already exists, and if so returns an error.
        Creates a temporary user entry with the auth provider and cli_id.

        Args:
            cli_id (str): The unique identifier from the CLI
            auth_provider (str): The authentication provider ('discord' or 'github')

        Raises:
            KernelBotError: If auth provider is invalid or cli_id already exists
        """
        if auth_provider not in ["discord", "github"]:
            raise Exception("Invalid auth provider")

        try:
            # Check if cli_id already exists
            self.cursor.execute(
                """
                SELECT 1 FROM leaderboard.user_info WHERE cli_id = %s
                """,
                (cli_id,),
            )

            if self.cursor.fetchone():
                raise Exception("CLI ID already exists")

            self.cursor.execute(
                """
                INSERT INTO leaderboard.user_info (id, user_name,
                cli_id, cli_auth_provider, cli_valid)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (f"temp_{cli_id}", f"temp_user_{cli_id}", cli_id, auth_provider, False),
            )

            self.connection.commit()
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.exception("Error initializing user from CLI with ID %s", cli_id, exc_info=e)
            raise KernelBotError("Error initializing user from CLI") from e

    def create_user_from_cli(self, user_id: str, user_name: str, cli_id: str, auth_provider: str):
        """
        Method to create a user from the CLI. Shouldn't be used for Discord.
        Validates that the user doesn't already have a valid row and that the user_id/user_name
        are temporary values that need to be updated.
        """
        try:
            self.cursor.execute(
                """
                SELECT 1 FROM leaderboard.user_info WHERE id = %s
                """,
                (user_id,),
            )
            if self.cursor.fetchone():
                raise Exception(
                    "User already has a valid account with this User ID."
                    "Please use the re-register command to re-authenticate."
                )

            self.cursor.execute(
                """
                SELECT cli_valid FROM leaderboard.user_info
                WHERE cli_id = %s AND cli_valid = TRUE AND cli_auth_provider = %s
                """,
                (cli_id, auth_provider),
            )

            if self.cursor.fetchone():
                raise Exception(
                    "User already has a valid account with this CLI ID."
                    "Please use the re-register command to re-authenticate."
                )

            self.cursor.execute(
                """
                UPDATE leaderboard.user_info
                SET id = %s, user_name = %s, cli_valid = TRUE, cli_auth_provider = %s
                WHERE cli_id = %s AND cli_valid = FALSE
                """,
                (user_id, user_name, auth_provider, cli_id),
            )

            if self.cursor.rowcount == 0:
                raise Exception("No temporary user found with this CLI ID. No effect.")

            self.connection.commit()
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.exception("Could not create/update user %s from CLI.", user_id, exc_info=e)
            raise KernelBotError("Database error while creating/updating user from CLI") from e

    def reset_user_from_cli(self, user_id: str, cli_id: str, auth_provider: str):
        try:
            self.cursor.execute(
                """
                SELECT 1 FROM leaderboard.user_info WHERE id = %s
                """,
                (user_id,),
            )
            if not self.cursor.fetchone():
                raise Exception(
                    "User not found. Please use the register command to create an account."
                )

            self.cursor.execute(
                """
                UPDATE leaderboard.user_info
                SET cli_id = %s, cli_auth_provider = %s, cli_valid = TRUE
                WHERE id = %s
                """,
                (cli_id, auth_provider, user_id),
            )

            self.connection.commit()
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.exception("Could not reset user %s from CLI.", user_id, exc_info=e)
            raise KernelBotError("Database error while resetting user from CLI") from e

    def cleanup_temp_users(self):
        try:
            self.cursor.execute(
                """
                DELETE FROM leaderboard.user_info WHERE cli_valid = FALSE and created_at <
                NOW() - INTERVAL '10 minutes' AND id LIKE 'temp_%' AND user_name LIKE 'temp_%'
                """
            )
            self.connection.commit()
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.exception("Could not cleanup temp users", exc_info=e)
            raise KernelBotError("Database error while cleaning up temp users") from e

    def validate_cli_id(self, cli_id: str) -> Optional[dict[str, str]]:
        """
        Validates a CLI ID and returns the associated user ID if valid.

        Args:
            cli_id (str): The CLI ID to validate.

        Returns:
            Optional[str]: The user ID if the CLI ID is valid, otherwise None.
        """
        try:
            self.cursor.execute(
                """
                SELECT id, user_name FROM leaderboard.user_info
                WHERE cli_id = %s AND cli_valid = TRUE
                """,
                (cli_id,),
            )
            result = self.cursor.fetchone()
            return {"user_id": result[0], "user_name": result[1]} if result else None
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.exception("Error validating CLI ID %s", cli_id, exc_info=e)
            raise KernelBotError("Error validating CLI ID") from e


class LeaderboardDoesNotExist(KernelBotError):
    def __init__(self, name: str):
        super().__init__(message=f"Leaderboard `{name}` does not exist.", code=404)
