########
# Evaluation scripts to run for leaderboard results
########

from pathlib import Path

py_eval = Path.read_text(Path(__file__).parent / "eval.py")
cu_eval = Path.read_text(Path(__file__).parent / "eval.cu")
