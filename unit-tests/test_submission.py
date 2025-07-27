import datetime
from unittest import mock

import pytest

from libkernelbot import submission
from libkernelbot.consts import RankCriterion
from libkernelbot.db_types import LeaderboardItem
from libkernelbot.utils import KernelBotError


@pytest.fixture
def mock_backend():
    """Create a mock backend with database context for testing."""
    backend = mock.Mock()
    backend.accepts_jobs = True

    # Mock database context manager
    db_context = mock.Mock()
    backend.db = db_context
    db_context.__enter__ = mock.Mock(return_value=db_context)
    db_context.__exit__ = mock.Mock(return_value=None)

    # Default mock responses
    mock_task = mock.Mock()
    db_context.get_leaderboard.return_value = {
        "task": mock_task,
        "secret_seed": 12345,
        "deadline": datetime.datetime.now() + datetime.timedelta(days=1),
        "name": "test_board",
    }
    db_context.get_leaderboard_gpu_types.return_value = ["A100", "V100"]

    return backend


def test_check_deadline():
    # Test valid deadline (future)
    future_deadline: LeaderboardItem = {
        "deadline": datetime.datetime.now() + datetime.timedelta(days=1),
        "name": "test",
    }
    submission.check_deadline(future_deadline)  # Should not raise

    # Test expired deadline
    past_deadline: LeaderboardItem = {
        "deadline": datetime.datetime.now() - datetime.timedelta(days=1),
        "name": "test",
    }

    with pytest.raises(
        KernelBotError, match=r"The deadline to submit to test has passed\.\nIt was.*and today is.*"
    ):
        submission.check_deadline(past_deadline)


def test_get_avail_gpus(mock_backend):
    db = mock_backend.db
    # Test with available GPUs
    result = submission.get_avail_gpus("test_board", db)
    assert result == ["A100", "V100"]

    # Test with no available GPUs
    db.get_leaderboard_gpu_types.return_value = []
    with pytest.raises(KernelBotError, match="No available GPUs"):
        submission.get_avail_gpus("test_board", db)


def test_get_popcorn_directives_valid():
    # Python comments with GPU and leaderboard directives
    code_py = """#!POPCORN gpu A100 V100
#!POPCORN leaderboard my_board
print("hello")"""
    result = submission._get_popcorn_directives(code_py)
    assert result == {"gpus": ["A100", "V100"], "leaderboard": "my_board"}

    # C++ comments
    code_cpp = """//!POPCORN gpu L4
//!POPCORN leaderboard cpp_board
int main() {}"""
    result = submission._get_popcorn_directives(code_cpp)
    assert result == {"gpus": ["L4"], "leaderboard": "cpp_board"}

    # no directives
    code_empty = """print("no directives")"""
    result = submission._get_popcorn_directives(code_empty)
    assert result == {"gpus": None, "leaderboard": None}

    # directives only in the first comment block
    code_mixed = """#!POPCORN leaderboard valid_board
print("code")
#!POPCORN gpu a100"""
    result = submission._get_popcorn_directives(code_mixed)
    assert result == {"gpus": None, "leaderboard": "valid_board"}

    # Only whitespace
    result = submission._get_popcorn_directives("   \n\t  \n")
    assert result == {"gpus": None, "leaderboard": None}

    # Case sensitivity
    code_case = """#!POPCORN GPUs a100 v100
#!POPCORN LEADERBOARD My_Board"""
    result = submission._get_popcorn_directives(code_case)
    # Assuming the function is case-insensitive for keywords but preserves case for values
    assert result == {"gpus": ["a100", "v100"], "leaderboard": "My_Board"}

    # Extra whitespace
    code_whitespace = """#!POPCORN  gpu   A100    V100  
#!POPCORN   leaderboard    my_board   """  # noqa: W291
    result = submission._get_popcorn_directives(code_whitespace)
    assert result == {"gpus": ["A100", "V100"], "leaderboard": "my_board"}

    # Single GPU
    code_single_gpu = """#!POPCORN gpu H100"""
    result = submission._get_popcorn_directives(code_single_gpu)
    assert result == {"gpus": ["H100"], "leaderboard": None}


def test_get_popcorn_directives_invalid():
    # Empty GPU list
    with pytest.raises(KernelBotError, match="!POPCORN directive missing argument: #!POPCORN gpu"):
        submission._get_popcorn_directives("#!POPCORN gpu")

    # Empty leaderboard but valid GPU
    code_empty_leaderboard = """#!POPCORN gpu A100
#!POPCORN leaderboard"""
    with pytest.raises(
        KernelBotError, match="!POPCORN directive missing argument: #!POPCORN leaderboard"
    ):
        submission._get_popcorn_directives(code_empty_leaderboard)

    # Invalid directive
    with pytest.raises(KernelBotError, match="Invalid !POPCORN directive: invalid_directive"):
        submission._get_popcorn_directives("#!POPCORN invalid_directive value")

    # Multiple leaderboard directives (last one wins or first one wins?)
    code_multiple_leaderboard = """#!POPCORN leaderboard first_board
#!POPCORN leaderboard second_board"""
    with pytest.raises(
        KernelBotError, match="Found multiple values for !POPCORN directive leaderboard"
    ):
        submission._get_popcorn_directives(code_multiple_leaderboard)


def test_handle_popcorn_directives():
    req = submission.SubmissionRequest(
        code="#!POPCORN leaderboard test_board",
        file_name="test.py",
        user_id=1,
        user_name="user",
        gpus=None,
        leaderboard=None,
    )

    # Directive sets leaderboard
    result = submission.handle_popcorn_directives(req)
    assert result.leaderboard == "test_board"

    # Consistent double-specification is fine
    req.leaderboard = "test_board"
    result = submission.handle_popcorn_directives(req)
    assert result.leaderboard == "test_board"

    # But inconsistent values are rejected
    req.leaderboard = "different_board"
    expected_error = (
        "Leaderboard name `different_board` specified in the "
        "command doesn't match the one in the submission script header `test_board`."
    )
    with pytest.raises(KernelBotError, match=expected_error):
        submission.handle_popcorn_directives(req)

    # Test missing leaderboard
    req.code = "print('no directive')"
    req.leaderboard = None
    with pytest.raises(KernelBotError, match="Missing leaderboard name"):
        submission.handle_popcorn_directives(req)


def test_prepare_submission_basic(mock_backend):
    req = submission.SubmissionRequest(
        code="#!POPCORN leaderboard test_board\nprint('hello world')",
        file_name="test.py",
        user_id=1,
        user_name="test_user",
        gpus=None,
        leaderboard=None,
    )

    result = submission.prepare_submission(req, mock_backend)

    assert isinstance(result, submission.ProcessedSubmissionRequest)
    assert result.leaderboard == "test_board"
    assert result.secret_seed == 12345
    assert result.gpus is None
    assert result.user_id == req.user_id
    assert result.user_name == req.user_name
    assert result.file_name == req.file_name
    assert result.code == req.code
    assert result.task == mock_backend.db.get_leaderboard()["task"]
    assert result.task_gpus == ["A100", "V100"]


def test_prepare_submission_explicit(mock_backend):
    req = submission.SubmissionRequest(
        code="print('hello world')",
        file_name="test.cu",
        user_id=2,
        user_name="test_user2",
        gpus=["A100"],
        leaderboard="test_board",
    )

    result = submission.prepare_submission(req, mock_backend)

    assert result.leaderboard == "test_board"
    assert result.gpus == ["A100"]
    assert result.file_name == "test.cu"
    assert result.task_gpus == ["A100", "V100"]


def test_prepare_submission_single_gpu_auto_assign(mock_backend):
    mock_backend.db.get_leaderboard_gpu_types.return_value = ["H100"]

    req = submission.SubmissionRequest(
        code="#!POPCORN leaderboard test_board\nint main() { return 0; }",
        file_name="test.cpp",
        user_id=3,
        user_name="test_user3",
        gpus=None,
        leaderboard=None,
    )

    result = submission.prepare_submission(req, mock_backend)

    assert result.gpus == ["H100"]
    assert result.task_gpus == ["H100"]


def test_prepare_submission_gpu_from_popcorn(mock_backend):
    # Customize the mock for multiple GPU scenario
    mock_backend.db.get_leaderboard_gpu_types.return_value = ["A100", "V100", "H100"]

    req = submission.SubmissionRequest(
        code="#!POPCORN gpu V100 H100\n#!POPCORN leaderboard test_board\nprint('test')",
        file_name="test.py",
        user_id=4,
        user_name="test_user4",
        gpus=None,
        leaderboard=None,
    )

    result = submission.prepare_submission(req, mock_backend)

    assert result.gpus == ["V100", "H100"]
    assert result.leaderboard == "test_board"


def test_prepare_submission_checks(mock_backend):
    req = submission.SubmissionRequest(
        code="#!POPCORN leaderboard test_board",
        file_name="test.py",
        user_id=1,
        user_name="user",
        gpus=None,
        leaderboard=None,
    )

    # backend not accepting jobs
    mock_backend.accepts_jobs = False
    with pytest.raises(KernelBotError, match="not accepting"):
        submission.prepare_submission(req, mock_backend)

    # profane filename
    mock_backend.accepts_jobs = True
    req.file_name = "this_file_can_fuck_right_off.py"
    with pytest.raises(KernelBotError, match="Please provide a non-rude filename"):
        submission.prepare_submission(req, mock_backend)

    # invalid file extension
    req.file_name = "test.txt"
    with pytest.raises(
        KernelBotError, match=r"Please provide a Python \(.py\) or CUDA \(.cu / .cuh / .cpp\) file"
    ):
        submission.prepare_submission(req, mock_backend)


def test_compute_score():
    mock_task = mock.Mock()
    mock_result = mock.Mock()

    # Test LAST ranking with single benchmark
    mock_task.ranking_by = RankCriterion.LAST
    mock_result.runs = {
        "leaderboard": mock.Mock(
            run=mock.Mock(
                result={
                    "benchmark-count": "1",
                    "benchmark.0.mean": "2000000000",  # 2 seconds in nanoseconds
                }
            )
        )
    }
    score = submission.compute_score(mock_result, mock_task, 1)
    assert score == 2.0

    # Test MEAN ranking with multiple benchmarks
    mock_task.ranking_by = RankCriterion.MEAN
    mock_result.runs["leaderboard"].run.result = {
        "benchmark-count": "2",
        "benchmark.0.mean": "1000000000",  # 1 second
        "benchmark.1.mean": "3000000000",  # 3 seconds
    }
    score = submission.compute_score(mock_result, mock_task, 1)
    assert score == 2.0  # (1 + 3) / 2

    # Test GEOM ranking with multiple benchmarks
    mock_task.ranking_by = RankCriterion.GEOM
    mock_result.runs["leaderboard"].run.result = {
        "benchmark-count": "2",
        "benchmark.0.mean": "4000000000",  # 4 seconds
        "benchmark.1.mean": "9000000000",  # 9 seconds
    }
    score = submission.compute_score(mock_result, mock_task, 1)
    assert score == 6.0  # sqrt(4 * 9)

    # Test LAST with multiple benchmarks (should raise error)
    mock_task.ranking_by = RankCriterion.LAST
    mock_result.runs["leaderboard"].run.result["benchmark-count"] = "2"
    with pytest.raises(KernelBotError, match="exactly one benchmark"):
        submission.compute_score(mock_result, mock_task, 1)
