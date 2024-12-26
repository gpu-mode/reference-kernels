"""
This migration adds a table to track the GPU types for a leaderboard,
and adds a column to track the GPU type for a run.
"""

from yoyo import step

__depends__ = {"20241221_01_54Oeg-rename-problem-table"}

steps = [
    step(
        """
         CREATE TABLE leaderboard.gpu_type (
             leaderboard_id INTEGER NOT NULL
                 REFERENCES leaderboard.leaderboard(id),
             gpu_type TEXT NOT NULL,
             PRIMARY KEY (leaderboard_id, gpu_type)
         )
         """,
        "DROP TABLE leaderboard.gpu_type",
    ),
    step(
        """
         ALTER TABLE leaderboard.runinfo ADD COLUMN gpu_type TEXT NOT NULL
         """,
        """
         ALTER TABLE leaderboard.runinfo DROP COLUMN gpu_type
         """,
    ),
]
