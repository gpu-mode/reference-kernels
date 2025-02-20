"""
remember forum_id for leaderboard:
    makes it much easier for us to keep track of the corresponding thread;
    we don't have to rely on name matching
"""

from yoyo import step

__depends__ = {"20250304_01_DzORz-collect-system-information-for-each-run"}

steps = [
    step(
        "ALTER TABLE leaderboard.leaderboard ADD COLUMN forum_id BIGINT NOT NULL DEFAULT -1",
    ),
    step("ALTER TABLE leaderboard.leaderboard ALTER COLUMN forum_id DROP DEFAULT;"),
]
