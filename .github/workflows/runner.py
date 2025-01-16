import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.append("src/discord-cluster-manager")

from run_eval import run_config

config = json.loads(Path("payload.json").read_text())
Path("payload.json").unlink()

result = asdict(run_config(config))

Path("result.json").write_text(json.dumps(result))
