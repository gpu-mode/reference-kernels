"""
user-info-add-cli-id
"""

from yoyo import step

__depends__ = {"20250329_01_7VjJJ-add-a-secret-seed-column"}

steps = [
    step(
        "ALTER TABLE leaderboard.user_info ADD COLUMN IF NOT EXISTS cli_id VARCHAR(255) DEFAULT NULL;"  # noqa: E501
    )
]
