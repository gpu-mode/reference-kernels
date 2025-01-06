"""
add-leaderboard-creator-id
"""

from yoyo import step

__depends__ = {"20241226_01_ZQSOK-add_gpu_type_to_submission"}

steps = [
    step(
        "ALTER TABLE leaderboard.leaderboard ADD COLUMN creator_id BIGINT NOT NULL DEFAULT -1",
    )
]
