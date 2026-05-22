#!/usr/bin/env python3
"""Merge partial chunk outputs (data/partials/*.json) into data/deadlines.json."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "deadlines.json"
PARTIALS = ROOT / "data" / "partials"


def main() -> int:
    parts = sorted(PARTIALS.glob("*.json"))
    if not parts:
        print("no partials found", file=sys.stderr)
        return 1

    merged = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": None,
        "conferences": {},
    }
    for p in parts:
        data = json.loads(p.read_text())
        merged["model"] = data.get("model") or merged["model"]
        merged["conferences"].update(data.get("conferences", {}))

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(merged, indent=2, sort_keys=True))
    print(f"Merged {len(parts)} chunk(s), {len(merged['conferences'])} conference(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
