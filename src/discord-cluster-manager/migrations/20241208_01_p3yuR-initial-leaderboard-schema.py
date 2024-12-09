"""
Creates the initial leaderboard schema, with tables problem, submission, and
runinfo.
"""

from yoyo import step

__depends__ = {}

steps = [
    step("CREATE SCHEMA leaderboard"),

    step("""
         CREATE TABLE IF NOT EXISTS leaderboard.problem (
             id SERIAL PRIMARY KEY,
             name TEXT UNIQUE NOT NULL,
             deadline TIMESTAMP WITH TIME ZONE NOT NULL,
             reference_code TEXT NOT NULL
         )
         """),

    step("""
         CREATE TABLE IF NOT EXISTS leaderboard.submission (
             id SERIAL PRIMARY KEY,
             problem_id INTEGER NOT NULL REFERENCES leaderboard.problem(id),
             name TEXT NOT NULL,
             user_id TEXT NOT NULL,
             code TEXT NOT NULL,
             submission_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
             score NUMERIC NOT NULL
         )
         """),

    step("CREATE INDEX ON leaderboard.submission (problem_id)"),

    step("""
         CREATE TABLE IF NOT EXISTS leaderboard.runinfo (
             id SERIAL PRIMARY KEY,
             submission_id INTEGER NOT NULL REFERENCES leaderboard.submission(id),
             stdout TEXT,
             ncu_output TEXT
         )
         """),

    step("CREATE INDEX ON leaderboard.runinfo (submission_id)")
]