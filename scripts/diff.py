#!/usr/bin/env python3
"""
Compare git HEAD's data/deadlines.json with the working-tree version,
append meaningful changes to CHANGELOG.md, and print a commit message.

Run AFTER extract.py and BEFORE `git add`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "deadlines.json"
CHANGELOG = ROOT / "CHANGELOG.md"


def git_show_head(path: Path) -> dict | None:
    rel = path.relative_to(ROOT)
    try:
        out = subprocess.check_output(
            ["git", "show", f"HEAD:{rel}"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return json.loads(out)
    except subprocess.CalledProcessError:
        return None


def deadline_key(d: dict) -> str:
    return f"{d.get('round') or ''}::{d['name']}"


def diff_conference(old: dict, new: dict) -> list[str]:
    """Return human-readable change lines for one conference."""
    lines = []
    name = new.get("name", new.get("id", "?"))
    old_dls = {deadline_key(d): d for d in (old.get("deadlines") or [])}
    new_dls = {deadline_key(d): d for d in (new.get("deadlines") or [])}

    for k, nd in new_dls.items():
        od = old_dls.get(k)
        label = f"{nd['name']}" + (f" ({nd['round']})" if nd.get("round") else "")
        if not od:
            lines.append(f"- **{name}**: added `{label}` → {nd['date_iso']}")
        elif od["date_iso"] != nd["date_iso"]:
            lines.append(
                f"- **{name}**: `{label}` moved {od['date_iso']} → {nd['date_iso']}"
            )

    for k, od in old_dls.items():
        if k not in new_dls:
            label = f"{od['name']}" + (f" ({od['round']})" if od.get("round") else "")
            lines.append(f"- **{name}**: removed `{label}` (was {od['date_iso']})")

    # Status transitions worth surfacing
    if old.get("status") == "ok" and new.get("status") != "ok":
        lines.append(f"- **{name}**: status changed to `{new.get('status')}` ({new.get('notes', '')})")

    return lines


def main() -> int:
    if not DATA_FILE.exists():
        print("no data file", file=sys.stderr)
        return 1
    new_data = json.loads(DATA_FILE.read_text())
    old_data = git_show_head(DATA_FILE) or {"conferences": {}}

    all_changes: list[str] = []
    for cid, new_conf in new_data["conferences"].items():
        old_conf = old_data.get("conferences", {}).get(cid, {})
        all_changes.extend(diff_conference(old_conf, new_conf))

    if not all_changes:
        print("NO_CHANGES")
        return 0

    # Append to CHANGELOG
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = f"## {today}\n\n" + "\n".join(all_changes) + "\n\n"
    existing = CHANGELOG.read_text() if CHANGELOG.exists() else "# Changelog\n\n"
    # Insert after the top "# Changelog" header
    if existing.startswith("# Changelog"):
        head, _, rest = existing.partition("\n\n")
        CHANGELOG.write_text(f"{head}\n\n{entry}{rest}")
    else:
        CHANGELOG.write_text(f"# Changelog\n\n{entry}{existing}")

    # Print commit message to stdout for the workflow to capture
    summary = f"Update deadlines ({len(all_changes)} change{'s' if len(all_changes) != 1 else ''})"
    body = "\n".join(all_changes[:20])
    print(f"{summary}\n\n{body}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
