"""
submission collection
see https://github.com/gpu-mode/discord-cluster-manager/pull/171
"""

from yoyo import step

__depends__ = {"20250202_01_YYS3Q-leaderboard-rename-reference-to-task"}

steps = [
    # Drop the old submissions table
    step("DROP TABLE IF EXISTS leaderboard.submission;"),
    step("DROP TABLE IF EXISTS leaderboard.code_files;"),
    step("DROP TABLE IF EXISTS leaderboard.runs;"),
    # create three new tables: One for deduplicating submitted code files,
    # one for the submission itself, and one for individual runs
    # The submission itself contains the code and the targeted leaderboard
    # in the future, we could, e.g., avoid starting a runner if we know that
    # the given code does not compile.
    step("""
         CREATE TABLE IF NOT EXISTS leaderboard.code_files (
             id SERIAL PRIMARY KEY,
             code TEXT NOT NULL,
             hash TEXT GENERATED ALWAYS AS (encode(sha256(code::bytea), 'hex')) STORED
         )
         """),
    step("""
         CREATE TABLE IF NOT EXISTS leaderboard.submission (
             id SERIAL PRIMARY KEY,
             leaderboard_id INTEGER NOT NULL REFERENCES leaderboard.leaderboard(id),
             file_name TEXT NOT NULL,
             user_id TEXT NOT NULL,
             code_id INTEGER NOT NULL REFERENCES leaderboard.code_files(id),
             submission_time TIMESTAMP WITH TIME ZONE NOT NULL,
             done BOOLEAN DEFAULT FALSE
         )
         """),
    # the runs themselves contain information about a particular execution of that code.
    # This includes start and end time
    # Note that `score` can be NULL for non-ranked submissions
    # `passed` indicates that the code compiled (if applicable), ran successfully,
    #          and passed all testing
    # `meta` contains all run "metadata", i.e., stdout and stderr, exit code etc
    # `result` is the actual result returned to our evaluation system
    step("""
         CREATE TABLE IF NOT EXISTS leaderboard.runs (
             id SERIAL PRIMARY KEY,
             submission_id INTEGER NOT NULL REFERENCES leaderboard.submission(id),
             start_time TIMESTAMP WITH TIME ZONE NOT NULL,
             end_time TIMESTAMP WITH TIME ZONE NOT NULL,
             mode TEXT NOT NULL,
             secret BOOLEAN NOT NULL,
             runner TEXT NOT NULL,
             score NUMERIC,
             passed BOOLEAN NOT NULL,
             compilation JSONB,
             meta JSONB,
             result JSONB
         )
         """),
]
