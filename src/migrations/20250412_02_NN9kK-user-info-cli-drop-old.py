"""
user-info-cli-drop-old
"""

from yoyo import step

__depends__ = {"20250412_01_l7Dra-user-info-fix-auth"}

steps = [
    step(
        """
        ALTER TABLE leaderboard.user_info
        ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();
        """
    )
]
