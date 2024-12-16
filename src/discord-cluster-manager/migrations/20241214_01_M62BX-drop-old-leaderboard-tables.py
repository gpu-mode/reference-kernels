"""
This migration drops tables in the public schema that we were previously
creating. The leaderboard tables are now in the leaderboard schema, so we dont'
need the tables in the public schema any more. I think the CASCADE clauses will
cause the sequences owned by the old tables to be dropped, but if not we'll need
another migration.
"""

from yoyo import step

__depends__ = {'20241208_01_p3yuR-initial-leaderboard-schema'}

steps = [
    step("DROP TABLE IF EXISTS public.leaderboard CASCADE"),
    step("DROP TABLE IF EXISTS public.submissions CASCADE"),
    step("DROP TABLE IF EXISTS public.runinfo CASCADE")
]
