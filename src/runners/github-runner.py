import base64
import json
import zlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from libkernelbot.run_eval import run_config

payload = Path("payload.json").read_text()
Path("payload.json").unlink()
payload = zlib.decompress(base64.b64decode(payload)).decode("utf-8")
config = json.loads(payload)

result = asdict(run_config(config))


# ensure valid serialization
def serialize(obj: object):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


Path("result.json").write_text(json.dumps(result, default=serialize))
