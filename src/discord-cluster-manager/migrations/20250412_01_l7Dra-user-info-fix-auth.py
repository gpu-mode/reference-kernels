"""
user-info-fix-auth
"""

from yoyo import step

__depends__ = {"20250406_01_ZXjWK-user-info-add-cli-id"}

steps = [
    step(
        """
        ALTER TABLE leaderboard.user_info
        ADD COLUMN IF NOT EXISTS cli_valid BOOLEAN DEFAULT FALSE;
        ALTER TABLE leaderboard.user_info
        ADD COLUMN IF NOT EXISTS cli_auth_provider VARCHAR(255) DEFAULT NULL;
        """
    )
]
