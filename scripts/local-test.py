import sys
from pathlib import Path

sys.path.append("src/discord-cluster-manager")

from leaderboard_eval import cu_eval
from run_eval import run_cuda_script

ref = Path("examples/identity_cuda/reference.cuh")
sub = Path("examples/identity_cuda/submission.cuh")

cout, score = run_cuda_script(
    {"eval.cu": cu_eval},
    {"reference.cuh": ref.read_text(), "submission.cuh": sub.read_text()},
    arch=None,
)
print(cout)
print(score)
exit(0 if score > 0 else 1)
