CREATE SCHEMA leaderboard;

CREATE TABLE IF NOT EXISTS leaderboard.problem (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    deadline TIMESTAMP WITH TIME ZONE NOT NULL,
    reference_code TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS leaderboard.submission (
    id SERIAL PRIMARY KEY,
    problem_id INTEGER NOT NULL REFERENCES leaderboard.problem(id),
    name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    code TEXT NOT NULL,
    submission_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    score NUMERIC NOT NULL
);

CREATE TABLE IF NOT EXISTS leaderboard.runinfo (
    id SERIAL PRIMARY KEY,
    submission_id INTEGER NOT NULL REFERENCES leaderboard.submission(id),
    stdout TEXT,
    ncu_output TEXT
);
