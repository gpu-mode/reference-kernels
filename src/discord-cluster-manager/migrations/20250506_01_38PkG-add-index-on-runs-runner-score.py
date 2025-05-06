"""
add index on runs(runner, score)
"""

from yoyo import step

__depends__ = {'20250412_02_NN9kK-user-info-cli-drop-old'}

steps = [
    step(
        "CREATE INDEX IF NOT EXISTS runs_runner_score_idx ON leaderboard.runs(runner, score)",
         "DROP INDEX IF EXISTS leaderboard.runs_runner_score_idx"
        )
]
