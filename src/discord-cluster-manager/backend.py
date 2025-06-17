import copy
import math
from typing import Optional

import env
from consts import GPU, GPU_TO_SM, RankCriterion, SubmissionMode
from launchers import Launcher
from leaderboard_db import LeaderboardDB
from report import RunProgressReporter, generate_report, make_short_report
from run_eval import FullResult
from task import LeaderboardTask, build_task_config
from utils import KernelBotError, setup_logging

logger = setup_logging(__name__)


class KernelBackend:
    def __init__(
        self,
        debug_mode=False,
    ):
        self.debug_mode = debug_mode
        self.db = LeaderboardDB(
            env.POSTGRES_HOST,
            env.POSTGRES_DATABASE,
            env.POSTGRES_USER,
            env.POSTGRES_PASSWORD,
            env.POSTGRES_PORT,
            url=env.DATABASE_URL,
            ssl_mode="require" if not env.DISABLE_SSL else "disable",
        )

        try:
            if not self.db.connect():
                logger.error("Could not connect to database, shutting down")
                exit(1)
        finally:
            self.db.disconnect()

        self.accepts_jobs = True
        self.launcher_map = {}

    def register_launcher(self, launcher: Launcher):
        for gpu in launcher.gpus:
            self.launcher_map[gpu.value] = launcher

    async def submit_leaderboard(  # noqa: C901
        self,
        submission_id: int,
        code: str,
        name: str,
        gpu_type: GPU,
        reporter: RunProgressReporter,
        task: LeaderboardTask,
        mode: SubmissionMode,
        seed: Optional[int],
    ) -> Optional[FullResult]:
        """
        Function invoked by `leaderboard_cog` to handle a leaderboard run.
        """
        if seed is not None:
            # careful, we've got a reference here
            # that is shared with the other run
            # invocations.
            task = copy.copy(task)
            task.seed = seed

        result = await self.handle_submission(
            gpu_type,
            reporter,
            code=code,
            name=name,
            task=task,
            mode=mode,
            submission_id=submission_id,
        )

        if result.success:
            score = None
            if (
                "leaderboard" in result.runs
                and result.runs["leaderboard"].run.success
                and result.runs["leaderboard"].run.passed
            ):
                score = 0.0
                num_benchmarks = int(result.runs["leaderboard"].run.result["benchmark-count"])
                if task.ranking_by == RankCriterion.LAST:
                    if num_benchmarks != 1:
                        logger.error(
                            "Ranked submission error for submission %d ranking_by is `last`, "
                            "but got %d benchmarks",
                            submission_id,
                            num_benchmarks,
                        )
                        raise KernelBotError(
                            f"Expected submission to have exactly one benchmark,"
                            f"got {num_benchmarks}."
                        )
                    score = float(result.runs["leaderboard"].run.result["benchmark.0.mean"]) / 1e9
                else:
                    scores = []
                    for i in range(num_benchmarks):
                        scores.append(
                            float(result.runs["leaderboard"].run.result[f"benchmark.{i}.mean"])
                            / 1e9
                        )
                    if task.ranking_by == RankCriterion.MEAN:
                        score = sum(scores) / len(scores)
                    elif task.ranking_by == RankCriterion.GEOM:
                        score = math.pow(math.prod(scores), 1.0 / num_benchmarks)

            # verifyruns uses a fake submission id of -1
            if submission_id != -1:
                with self.db as db:
                    for key, value in result.runs.items():
                        db.create_submission_run(
                            submission_id,
                            value.start,
                            value.end,
                            mode=key,
                            runner=gpu_type.name,
                            score=None if key != "leaderboard" else score,
                            secret=mode == SubmissionMode.PRIVATE,
                            compilation=value.compilation,
                            result=value.run,
                            system=result.system,
                        )

        return result

    async def handle_submission(
        self,
        gpu_type: GPU,
        reporter: RunProgressReporter,
        code: str,
        name: str,
        task: Optional[LeaderboardTask],
        mode: SubmissionMode,
        submission_id: int = -1,
    ) -> Optional[FullResult]:
        """
        Generic function to handle code submissions.
        Args:
            gpu_type: Which GPU to run on.
            code: Submitted code
            name: File name of the submission; used to infer code's language
            task: Task specification, of provided
            submission_id: ID of the submission, only used for display purposes

        Returns:
            if successful, returns the result of the run.
        """
        launcher = self.launcher_map[gpu_type.value]
        config = build_task_config(
            task=task, submission_content=code, arch=self._get_arch(gpu_type), mode=mode
        )

        logger.info("submitting task to runner %s", launcher.name)

        result = await launcher.run_submission(config, gpu_type, reporter)

        if not result.success:
            await reporter.update_title(reporter.title + " ❌ failure")
            await reporter.push(result.error)
            return result
        else:
            await reporter.update_title(reporter.title + " ✅ success")

        short_report = make_short_report(
            result.runs, full=mode in [SubmissionMode.PRIVATE, SubmissionMode.LEADERBOARD]
        )
        await reporter.push(short_report)
        if mode != SubmissionMode.PRIVATE:
            try:
                # does the last message of the short report start with ✅ or ❌?
                verdict = short_report[-1][0]
                id_str = f"{verdict}" if submission_id == -1 else f"{verdict} #{submission_id}"
                await reporter.display_report(
                    f"{id_str} {name} on {gpu_type.name} ({launcher.name})",
                    generate_report(result),
                )
            except Exception as E:
                logger.error("Error generating report. Result: %s", result, exc_info=E)
                raise

        return result

    def _get_arch(self, gpu_type: GPU):
        return GPU_TO_SM[gpu_type.name]
