import copy
import dataclasses
import datetime
import subprocess
import time

import pytest
from test_report import sample_compile_result, sample_run_result, sample_system_info
from test_task import task_directory

from libkernelbot import leaderboard_db
from libkernelbot.utils import KernelBotError

DATABASE_URL = "postgresql://postgres:postgres@localhost:5433/clusterdev"


@pytest.fixture(scope="module")
def docker_compose():
    """Start a test database and run migrations"""
    subprocess.check_call(["docker", "compose", "-f", "docker-compose.test.yml", "up", "-d"])

    try:
        # Wait for migrations to finish
        while True:
            result = subprocess.run(
                ["docker", "compose", "-f", "docker-compose.test.yml", "ps", "-q", "migrate-test"],
                capture_output=True,
                text=True,
            )

            if not result.stdout.strip():  # Container no longer exists
                break
            time.sleep(1)

        # Check if migrations succeeded
        logs = subprocess.run(
            ["docker", "compose", "-f", "docker-compose.test.yml", "logs", "migrate-test"],
            capture_output=True,
            text=True,
        )

        if "error" in logs.stdout.lower():
            raise Exception(f"Migrations failed: {logs.stdout}")

        yield leaderboard_db.LeaderboardDB(
            host="",
            database="",
            port="",
            user="",
            password="",
            url=DATABASE_URL,
            ssl_mode="disable",
        )
    finally:
        subprocess.run(["docker", "compose", "-f", "docker-compose.test.yml", "down", "-v"])


def _nuke_contents(db):
    db.cursor.execute(
        "TRUNCATE leaderboard.code_files, leaderboard.submission, leaderboard.runs, "
        "leaderboard.leaderboard, leaderboard.user_info, leaderboard.templates, "
        "leaderboard.gpu_type RESTART IDENTITY CASCADE"
    )
    db.connection.commit()


@pytest.fixture()
def database(docker_compose):
    with docker_compose as db:
        _nuke_contents(db)
    yield docker_compose
    with docker_compose as db:
        _nuke_contents(db)


def _submit_leaderboard(database, task_directory):
    """
    Creates a leaderboard called 'submit-leaderboard' and returns its ID.
    """
    from libkernelbot.task import make_task_definition

    definition = make_task_definition(task_directory / "task.yml")
    deadline = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(days=1)

    with database as db:
        return db.create_leaderboard(
            name="submit-leaderboard",
            deadline=deadline,
            definition=definition,
            creator_id=1,
            forum_id=5,
            gpu_types=["A100", "H100"],
        )


@pytest.fixture()
def submit_leaderboard(database, task_directory):
    return _submit_leaderboard(database, task_directory)


def _create_submission_run(
    db: leaderboard_db.LeaderboardDB,
    submission: int,
    *,
    start=None,
    end=None,
    mode="leaderboard",
    secret=False,
    runner="A100",
    score=None,
    compilation=None,
    system=None,
    result=None,
):
    """Creates a submission run with suitable default values"""
    db.create_submission_run(
        submission,
        start=start or datetime.datetime.now(tz=datetime.timezone.utc),
        end=end
        or (datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=10)),
        mode=mode,
        secret=secret,
        runner=runner,
        score=score,
        compilation=compilation or sample_compile_result(),
        result=result or sample_run_result(),
        system=system or sample_system_info(),
    )


def test_empty_db(database):
    expected_error = "Leaderboard `does-not-exist` does not exist."
    with database as db:
        with pytest.raises(leaderboard_db.LeaderboardDoesNotExist, match=expected_error):
            db.get_leaderboard("does-not-exist")
        with pytest.raises(leaderboard_db.LeaderboardDoesNotExist, match=expected_error):
            db.get_leaderboard_templates("does-not-exist")
        with pytest.raises(leaderboard_db.LeaderboardDoesNotExist, match=expected_error):
            db.get_leaderboard_gpu_types("does-not-exist")
        with pytest.raises(leaderboard_db.LeaderboardDoesNotExist, match=expected_error):
            db.get_leaderboard_submissions("does-not-exist", "A100", "5", 100)
        with pytest.raises(leaderboard_db.LeaderboardDoesNotExist, match=expected_error):
            db.get_leaderboard_submission_count("does-not-exist", "A100", "5")
        assert db.get_leaderboards() == []
        assert db.get_leaderboard_names() == []
        assert db.get_submission_by_id(0) is None
        assert db.get_user_from_id("0") is None


def test_nested_enter(database):
    with database as db_outer:
        with db_outer as db_inner:
            assert db_inner.get_leaderboards() == []


def test_leaderboard_basics(database, task_directory):
    """
    This test creates an empty leaderboard and checks its properties.
    """
    from libkernelbot.task import make_task_definition

    definition = make_task_definition(task_directory / "task.yml")

    deadline = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(days=1)

    with database as db:
        db.create_leaderboard(
            name="test-leaderboard",
            deadline=deadline,
            definition=definition,
            creator_id=1,
            forum_id=5,
            gpu_types=["A100", "H100"],
        )

        assert db.get_leaderboard_names() == ["test-leaderboard"]
        lb = db.get_leaderboard("test-leaderboard")

        assert lb["name"] == "test-leaderboard"
        assert lb["creator_id"] == 1
        assert lb["deadline"] == deadline
        assert lb["description"] == definition.description
        assert lb["task"] == definition.task
        assert lb["gpu_types"] == ["A100", "H100"]
        assert lb["forum_id"] == 5
        assert lb["id"] == db.get_leaderboard_id("test-leaderboard")
        assert isinstance(lb["secret_seed"], int)

        assert db.get_leaderboards() == [lb]

        assert db.get_leaderboard_templates("test-leaderboard") == {
            "Python": "# Python template",
            "CUDA": "// CUDA template",
        }
        assert db.get_leaderboard_gpu_types("test-leaderboard") == ["A100", "H100"]
        assert db.get_leaderboard_submissions("test-leaderboard", "A100", "5", 100) == []
        assert db.get_leaderboard_submission_count("test-leaderboard", "A100", "5") == 0

        with pytest.raises(
            KernelBotError, match="Invalid GPU type 'A99' for leaderboard 'test-leaderboard'"
        ):
            assert db.get_leaderboard_submissions("test-leaderboard", "A99", "5", 100) == []

        with pytest.raises(
            KernelBotError, match="Invalid GPU type 'A99' for leaderboard 'test-leaderboard'"
        ):
            assert db.get_leaderboard_submission_count("test-leaderboard", "A99", "5") == 0


def test_recreate_leaderboard(database, task_directory):
    _submit_leaderboard(database, task_directory)
    with pytest.raises(
        KernelBotError,
        match="Error: Tried to create a leaderboard 'submit-leaderboard' that already exists.",
    ):
        _submit_leaderboard(database, task_directory)


def test_expired_leaderboard(database, task_directory):
    from libkernelbot.task import make_task_definition

    definition = make_task_definition(task_directory / "task.yml")
    deadline = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=1)

    _submit_leaderboard(database, task_directory)
    with database as db:
        db.create_leaderboard(
            name="other-leaderboard",
            deadline=deadline,
            definition=definition,
            creator_id=1,
            forum_id=5,
            gpu_types=["A100", "H100"],
        )

        assert len(db.get_leaderboard_names()) == 2
        assert db.get_leaderboard_names(active_only=True) == ["submit-leaderboard"]


def test_leaderboard_submission_basic(database, submit_leaderboard):
    """
    This test creates a leaderboard, adds a submission and a few runs, then checks query results.
    """
    submit_time = datetime.datetime.now(tz=datetime.timezone.utc)

    # we used to have problems with literal \n in source files, so let's test that here
    dangerous_code = r"'python string with\nspecial\tcharacters'"

    with database as db:
        sub_id = db.create_submission(
            "submit-leaderboard", "submission.py", 5, dangerous_code, submit_time, user_name="user"
        )

        # check the raw submission
        submission = db.get_submission_by_id(sub_id)
        assert submission["submission_id"] == sub_id
        assert submission["leaderboard_id"] == db.get_leaderboard_id("submit-leaderboard")
        assert submission["leaderboard_name"] == "submit-leaderboard"
        assert submission["file_name"] == "submission.py"
        assert submission["user_id"] == "5"  # TODO str or int?
        assert submission["submission_time"] == submit_time
        assert submission["done"] is False
        assert submission["code"] == dangerous_code
        assert submission["runs"] == []

    # add a submission run
    run_result = sample_run_result()
    with database as db:
        end_time = submit_time + datetime.timedelta(seconds=10)
        db.create_submission_run(
            sub_id,
            submit_time,
            end_time,
            mode="test",
            secret=False,
            runner="A100",
            score=None,
            compilation=None,
            result=run_result,
            system=sample_system_info(),
        )
        # run ends after the contest deadline; this is valid
        end_time_2 = submit_time + datetime.timedelta(days=1, hours=1)
        db.create_submission_run(
            sub_id,
            submit_time,
            end_time_2,
            mode="leaderboard",
            secret=True,
            runner="H100",
            score=5.5,
            compilation=sample_compile_result(),
            result=run_result,
            system=sample_system_info(),
        )

        expected_meta = {
            k: getattr(run_result, k)
            for k in ("stdout", "stderr", "success", "exit_code", "command", "duration")
        }

        submission = db.get_submission_by_id(sub_id)

        assert len(submission["runs"]) == 2
        for run in submission["runs"]:
            if run["mode"] == "test":
                assert run["start_time"] == submit_time
                assert run["end_time"] == end_time
                assert run["secret"] is False
                assert run["runner"] == "A100"
                assert run["score"] is None
                assert run["compilation"] is None
                assert run["passed"] is True
                assert run["meta"] == expected_meta
                assert run["result"] == run_result.result
                assert run["system"] == dataclasses.asdict(sample_system_info())
            elif run["mode"] == "leaderboard":
                assert run["start_time"] == submit_time
                assert run["end_time"] == end_time_2
                assert run["secret"] is True
                assert run["runner"] == "H100"
                assert run["score"] == 5.5
                assert run["passed"] is True
                assert run["compilation"] == dataclasses.asdict(sample_compile_result())
                assert run["meta"] == expected_meta
                assert run["result"] == run_result.result
                assert run["system"] == dataclasses.asdict(sample_system_info())

        db.mark_submission_done(sub_id)

        with pytest.raises(KernelBotError):
            _create_submission_run(db, sub_id)


def test_leaderboard_submission_count(database, submit_leaderboard):
    """Check submission counting logic"""
    submit_time = datetime.datetime.now(tz=datetime.timezone.utc)

    # we used to have problems with literal \n in source files, so let's test that here
    dangerous_code = r"'python string with\nspecial\tcharacters'"

    with database as db:
        sub_id = db.create_submission(
            "submit-leaderboard", "submission.py", 5, dangerous_code, submit_time, user_name="user"
        )
        _create_submission_run(db, sub_id, mode="test", secret=False, runner="A100")
        _create_submission_run(
            db, sub_id, mode="leaderboard", secret=True, runner="H100", score=5.5
        )
        _create_submission_run(
            db, sub_id, mode="leaderboard", secret=False, runner="A100", score=1.5
        )
        submission = db.get_submission_by_id(sub_id)

        assert len(submission["runs"]) == 3

        db.mark_submission_done(sub_id)
    with database as db:
        # H100: secret, not counted
        assert db.get_leaderboard_submission_count("submit-leaderboard", "H100") == 0
        # A100: only one of the two submissions has a score assigned
        assert db.get_leaderboard_submission_count("submit-leaderboard", "A100") == 1
        assert db.get_leaderboard_submission_count("submit-leaderboard", "A100", "5") == 1
        assert db.get_leaderboard_submission_count("submit-leaderboard", "H100", "6") == 0


def test_leaderboard_submission_ranked(database, submit_leaderboard):
    """Check submission counting logic"""
    submit_time = datetime.datetime.now(tz=datetime.timezone.utc)

    # we used to have problems with literal \n in source files, so let's test that here
    dangerous_code = r"'python string with\nspecial\tcharacters'"

    with database as db:
        sub_id = db.create_submission(
            "submit-leaderboard", "submission.py", 5, dangerous_code, submit_time, user_name="user"
        )
        _create_submission_run(db, sub_id, mode="leaderboard", runner="A100", score=5.5)
        db.mark_submission_done(sub_id)

        sub_id = db.create_submission(
            "submit-leaderboard", "submission.py", 5, dangerous_code, submit_time, user_name="user"
        )
        _create_submission_run(db, sub_id, mode="leaderboard", runner="A100", score=4.5)
        db.mark_submission_done(sub_id)

        sub_id = db.create_submission(
            "submit-leaderboard", "submission.py", 5, dangerous_code, submit_time, user_name="user"
        )
        _create_submission_run(db, sub_id, mode="leaderboard", runner="A100", score=5.0)
        db.mark_submission_done(sub_id)

        sub_id = db.create_submission(
            "submit-leaderboard", "submission.py", 6, dangerous_code, submit_time, user_name="user"
        )
        _create_submission_run(db, sub_id, mode="leaderboard", runner="A100", score=8.0)
        db.mark_submission_done(sub_id)

        sub_id = db.create_submission(
            "submit-leaderboard", "submission.py", 6, dangerous_code, submit_time, user_name="user"
        )
        _create_submission_run(db, sub_id, mode="leaderboard", runner="H100", score=2.0)
        db.mark_submission_done(sub_id)

    with database as db:
        ranked_sub = db.get_leaderboard_submissions("submit-leaderboard", "A100", None)
        from decimal import Decimal

        assert ranked_sub == [
            {
                "gpu_type": "A100",
                "leaderboard_name": "submit-leaderboard",
                "rank": 1,
                "submission_id": 2,
                "submission_name": "submission.py",
                "submission_score": Decimal("4.5"),
                "submission_time": submit_time,
                "user_id": "5",
                "user_name": "user",
            },
            {
                "gpu_type": "A100",
                "leaderboard_name": "submit-leaderboard",
                "rank": 2,
                "submission_id": 4,
                "submission_name": "submission.py",
                "submission_score": Decimal("8.0"),
                "submission_time": submit_time,
                "user_id": "6",
                "user_name": "user",
            },
        ]


def test_leaderboard_submission_deduplication(database, submit_leaderboard):
    """validate that identical submission codes are added just once"""
    with database as db:
        db.create_submission(
            "submit-leaderboard",
            "submission.py",
            5,
            "pass",
            datetime.datetime.now(),
            user_name="user",
        )
        db.create_submission(
            "submit-leaderboard", "other.py", 6, "pass", datetime.datetime.now(), user_name="other"
        )

        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.code_files")
        assert db.cursor.fetchone()[0] == 1


def test_leaderboard_submission_delete(database, submit_leaderboard):
    with database as db:
        sub_id = db.create_submission(
            "submit-leaderboard",
            "submission.py",
            5,
            "pass",
            datetime.datetime.now(),
            user_name="user",
        )
        other_sub = db.create_submission(
            "submit-leaderboard",
            "submission.py",
            5,
            "different",
            datetime.datetime.now(),
            user_name="user",
        )

        _create_submission_run(db, sub_id, mode="leaderboard", secret=False, runner="A100", score=5)
        _create_submission_run(db, sub_id, mode="leaderboard", secret=True, runner="A100", score=5)
        _create_submission_run(
            db, other_sub, mode="leaderboard", secret=False, runner="A100", score=5
        )
        db.mark_submission_done(sub_id)

        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.runs")
        assert db.cursor.fetchone()[0] == 3

        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.submission")
        assert db.cursor.fetchone()[0] == 2

        # ok, now delete
        db.delete_submission(sub_id)
        assert db.get_submission_by_id(sub_id) is None
        assert db.get_submission_by_id(other_sub) is not None

        # run and submission are deleted
        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.runs")
        assert db.cursor.fetchone()[0] == 1

        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.submission")
        assert db.cursor.fetchone()[0] == 1

        # but the code file remains
        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.code_files")
        assert db.cursor.fetchone()[0] == 2


def test_delete_leaderboard(database, submit_leaderboard):
    with database as db:
        db.delete_leaderboard("submit-leaderboard")
        assert db.get_leaderboard_names() == []


def test_delete_leaderboard_with_runs(database, submit_leaderboard):
    with database as db:
        db.create_submission(
            "submit-leaderboard",
            "submission.py",
            5,
            "pass",
            datetime.datetime.now(),
            user_name="user",
        )

        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.templates")
        assert db.cursor.fetchone()[0] > 0

        with pytest.raises(
            KernelBotError,
            match="Could not delete leaderboard `submit-leaderboard` with existing submissions.",
        ):
            db.delete_leaderboard("submit-leaderboard")

        # nothing was deleted
        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.templates")
        assert db.cursor.fetchone()[0] > 0
        assert db.get_leaderboard_names() == ["submit-leaderboard"]

        db.delete_leaderboard("submit-leaderboard", force=True)
        assert db.get_leaderboard_names() == []
        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.submission")
        assert db.cursor.fetchone()[0] == 0

        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.templates")
        assert db.cursor.fetchone()[0] == 0


def test_leaderboard_update(database, task_directory):
    from libkernelbot.task import make_task_definition

    definition = make_task_definition(task_directory / "task.yml")

    deadline = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(days=1)
    new_deadline = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(days=2)

    new_def = copy.deepcopy(definition)
    new_def.description = "new description"
    new_def.task.test_timeout = 14532
    new_def.templates["CUDA"] = "// new CUDA template"

    with database as db:
        # create initial leaderboard
        db.create_leaderboard(
            name="test-leaderboard",
            deadline=deadline,
            definition=definition,
            creator_id=1,
            forum_id=5,
            gpu_types=["A100", "H100"],
        )

        # update deadline
        db.update_leaderboard("test-leaderboard", new_deadline, new_def)
        updated_lb = db.get_leaderboard("test-leaderboard")
        assert updated_lb["deadline"] == new_deadline
        assert updated_lb["description"] == "new description"
        assert updated_lb["task"] == new_def.task

        assert db.get_leaderboard_templates("test-leaderboard") == {
            "CUDA": "// new CUDA template",
            "Python": "# Python template",
        }


def test_generate_stats(database, submit_leaderboard):
    with database as db:
        start = datetime.datetime.now(tz=datetime.timezone.utc)
        sub_id = db.create_submission(
            "submit-leaderboard", "submission.py", 5, "pass", start, user_name="user"
        )
        _create_submission_run(
            db,
            sub_id,
            start=start + datetime.timedelta(seconds=10),
            end=start + datetime.timedelta(seconds=20),
            mode="leaderboard",
            secret=False,
            runner="A100",
            score=5,
        )
        _create_submission_run(
            db,
            sub_id,
            start=start + datetime.timedelta(seconds=20),
            end=start + datetime.timedelta(seconds=30),
            mode="leaderboard",
            secret=True,
            runner="A100",
            score=6,
        )
        _create_submission_run(
            db,
            sub_id,
            start=start,
            end=start + datetime.timedelta(seconds=15),
            mode="leaderboard",
            secret=False,
            runner="A100",
            score=4,
        )
        db.mark_submission_done(sub_id)

        assert db.generate_stats(False) == {
            "avg_delay.A100": datetime.timedelta(seconds=10),
            "max_delay.A100": datetime.timedelta(seconds=20),
            "num_run.A100": 3,
            "num_submissions": 1,
            "num_unique_codes": 1,
            "num_users": 1,
            "runs_passed.A100": 3,
            "runs_scored.A100": 3,
            "runs_secret.A100": 1,
            "sub_waiting": 0,
            "total_runtime.A100": datetime.timedelta(seconds=35),
        }


# this is her just to make ruff leave pytest fixtures alone
__all__ = [task_directory]
