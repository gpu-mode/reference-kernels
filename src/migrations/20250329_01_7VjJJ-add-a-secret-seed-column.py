"""
Add a secret seed column.
Note: double epsilon is eps=2.22e-16, so 1/eps = 4504504504504504 >> 2^31 = 2147483648,
so we should get reasonably uniform integers here
"""

from yoyo import step

__depends__ = {"20250316_01_5oMi3-remember-forum-id"}

steps = [
    step(
        "ALTER TABLE leaderboard.leaderboard ADD COLUMN secret_seed BIGINT NOT NULL "
        "DEFAULT FLOOR(RANDOM() * 2147483648) ",
    ),
]
