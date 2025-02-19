import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.append("src/discord-cluster-manager")

from run_eval import run_config

config = json.loads(Path("payload.json").read_text())
Path("payload.json").unlink()

result = asdict(run_config(config))


# ensure valid serialization
def serialize(obj: object):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


Path("result.json").write_text(json.dumps(result, default=serialize))
