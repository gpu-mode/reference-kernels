# This file provides TypeDict definitions for the return types we get from database queries
import datetime
from typing import TYPE_CHECKING, List, NotRequired, Optional, TypedDict

if TYPE_CHECKING:
    from libkernelbot.task import LeaderboardTask


class LeaderboardItem(TypedDict):
    id: int
    name: str
    creator_id: int
    deadline: datetime.datetime
    description: str
    task: "LeaderboardTask"
    gpu_types: List[str]
    forum_id: int
    secret_seed: NotRequired[int]


class LeaderboardRankedEntry(TypedDict):
    submission_id: int
    rank: int
    submission_name: str
    submission_time: datetime.datetime
    submission_score: float
    leaderboard_name: str
    user_id: int
    user_name: str
    gpu_type: str


class RunItem(TypedDict):
    start_time: datetime.datetime
    end_time: datetime.datetime
    mode: str
    secret: bool
    runner: str
    score: Optional[float]
    passed: bool
    compilation: dict
    meta: dict
    result: dict
    system: dict


class SubmissionItem(TypedDict):
    submission_id: int
    leaderboard_id: int
    leaderboard_name: str
    file_name: str
    user_id: int
    submission_time: datetime.datetime
    done: bool
    code: str
    runs: List[RunItem]


__all__ = [LeaderboardItem, LeaderboardRankedEntry, RunItem, SubmissionItem]
