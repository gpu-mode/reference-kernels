"""
This migration renames the table leaderboard.problem to leaderboard.leaderboard,
and renames associated foreign keys, indexes, and constraints.
"""

from yoyo import step

__depends__ = {"20241214_01_M62BX-drop-old-leaderboard-tables"}

steps = [
    # Rename leaderboard.problem to leaderboard.leaderboard:
    step("ALTER TABLE leaderboard.problem RENAME TO leaderboard"),
    step("""
         ALTER SEQUENCE leaderboard.problem_id_seq
         RENAME TO leaderboard_id_seq
         """),
    step("ALTER INDEX leaderboard.problem_pkey RENAME TO leaderboard_pkey"),
    step("""
         ALTER TABLE leaderboard.leaderboard
         RENAME CONSTRAINT problem_name_key
         TO leaderboard_name_key
         """),
    # Rename a column in leaderboard.submission:
    step("""
         ALTER TABLE leaderboard.submission
         RENAME COLUMN problem_id
         TO leaderboard_id
         """),
    step("""
         ALTER TABLE leaderboard.submission
         RENAME CONSTRAINT submission_problem_id_fkey
         TO submission_leaderboard_id_fkey
         """),
    # We need to update the name for the index
    # leaderboard.submission_problem_id_idx. We can't just rename the index,
    # because the index definition refers to the old column name, problem_id.
    # This might be confusing, so we drop and re-create the index:
    step("DROP INDEX leaderboard.submission_problem_id_idx"),
    step("""
         CREATE INDEX submission_leaderboard_id_idx
         ON leaderboard.submission (leaderboard_id)
         """),
]
