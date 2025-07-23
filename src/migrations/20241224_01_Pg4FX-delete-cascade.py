"""
This migration changes all existing foreign key constraints to be 'on delete
cascade'.
"""

from yoyo import step

__depends__ = {"20241222_01_ELxU5-add-gpu-types"}

steps = [
    # submission table
    step(
        """
        ALTER TABLE leaderboard.submission
        DROP CONSTRAINT submission_leaderboard_id_fkey
        """
    ),
    step(
        """
        ALTER TABLE leaderboard.submission
        ADD CONSTRAINT submission_leaderboard_id_fkey
        FOREIGN KEY (leaderboard_id)
        REFERENCES leaderboard.leaderboard(id)
        ON DELETE CASCADE
        """
    ),
    # runinfo table
    step(
        """
        ALTER TABLE leaderboard.runinfo
        DROP CONSTRAINT runinfo_submission_id_fkey
        """
    ),
    step(
        """
        ALTER TABLE leaderboard.runinfo
        ADD CONSTRAINT runinfo_submission_id_fkey
        FOREIGN KEY (submission_id)
        REFERENCES leaderboard.submission(id)
        ON DELETE CASCADE
        """
    ),
    # gpu_type table
    step(
        """
        ALTER TABLE leaderboard.gpu_type
        DROP CONSTRAINT gpu_type_leaderboard_id_fkey
        """
    ),
    step(
        """
        ALTER TABLE leaderboard.gpu_type
        ADD CONSTRAINT gpu_type_leaderboard_id_fkey
        FOREIGN KEY (leaderboard_id)
        REFERENCES leaderboard.leaderboard(id)
        ON DELETE CASCADE
        """
    ),
]
