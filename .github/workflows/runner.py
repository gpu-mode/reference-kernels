import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.append("src/discord-cluster-manager")

from leaderboard_eval import cu_eval, py_eval
from run_eval import run_cuda_script, run_pytorch_script

config = json.loads(Path("payload.json").read_text())  # type: dict
Path("payload.json").unlink()

if config["lang"] == "cu":
    comp, run = run_cuda_script(
        config.get("eval.cu", cu_eval),
        config.get("reference.cuh", None),
        config.get("submission.cuh", None),
        arch=None,
    )
    result = {"compile": asdict(comp), "run": asdict(run)}
else:
    run = run_pytorch_script(
        config.get("eval.py", py_eval), config.get("reference.py", None), config.get("submission.py", None), arch=None
    )
    result = {"run": asdict(run)}

Path("result.json").write_text(json.dumps(result))
