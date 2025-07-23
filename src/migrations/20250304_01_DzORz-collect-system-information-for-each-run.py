"""
Collect system information for each run.
"""

from yoyo import step

__depends__ = {"20250228_01_9ANYn-submission-add-user-name"}

steps = [
    step(
        "ALTER TABLE leaderboard.runs ADD COLUMN system_info JSONB NOT NULL DEFAULT '{}'::jsonb;",
    ),
    step("ALTER TABLE leaderboard.runs ALTER COLUMN system_info DROP DEFAULT;"),
]
