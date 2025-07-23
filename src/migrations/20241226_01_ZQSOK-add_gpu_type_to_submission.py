"""
This migration adds a score column to runinfo, and drops the score column in
submission.
"""

from yoyo import step

__depends__ = {"20241224_01_Pg4FX-delete-cascade"}

steps = [
    step("DROP TABLE leaderboard.runinfo"),
    step("""
         ALTER TABLE leaderboard.submission
         ADD COLUMN gpu_type TEXT NOT NULL DEFAULT 'nvidia'
         """),
    step("ALTER TABLE leaderboard.submission ADD COLUMN stdout TEXT"),
    step("ALTER TABLE leaderboard.submission ADD COLUMN profiler_output TEXT"),
]
