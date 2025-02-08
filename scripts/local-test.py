import pprint
import sys
from pathlib import Path

sys.path.append("src/discord-cluster-manager")

from run_eval import run_cuda_script

ref = Path("examples/identity_cuda/reference.cuh")
sub = Path("examples/identity_cuda/submission.cu")
util = Path("examples/identity_cuda/utils.h")
task = Path("examples/identity_cuda/task.h")

result = run_cuda_script(
    {
        "eval.cu": Path("examples/identity_cuda/eval.cu").read_text(),
        "submission.cu": sub.read_text(),
    },
    {"reference.cuh": ref.read_text(), "utils.h": util.read_text(), "task.h": task.read_text()},
    arch=None,
    tests="size: 128; seed: 45\nsize: 512; seed: 123",
    mode="test",
)

pprint.pprint(result)
