#!/usr/bin/env python3
"""
Nightly deadline extractor.

Flow per conference:
  1. Fetch seed_url. If it looks like a hub page (not the actual CFP),
     ask the LLM to find the current CFP link, then fetch that.
  2. Strip page to clean text via trafilatura.
  3. Ask LLM (with JSON schema) to extract deadlines.
  4. Merge with previous state, preserving history.

Outputs: data/deadlines.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "deadlines.json"
CONFIG_FILE = ROOT / "conferences.yaml"

# GitHub Models. Token comes from GITHUB_TOKEN in Actions (with models:read).
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=os.environ["GITHUB_TOKEN"],
)

# Keep page text well under the 8K input token limit on free GitHub Models tier.
# ~4 chars/token, leave room for prompt + schema.
MAX_PAGE_CHARS = 20_000


# ---------- Schemas for structured output ----------

DEADLINE_EXTRACTION_SCHEMA = {
    "name": "deadline_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "is_cfp_page": {
                "type": "boolean",
                "description": "True if this page actually describes a call for papers with deadlines.",
            },
            "conference_year": {
                "type": ["integer", "null"],
                "description": "Year of the conference (not necessarily the deadline year).",
            },
            "is_past": {
                "type": "boolean",
                "description": "True if all deadlines on this page have already passed.",
            },
            "deadlines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "e.g., 'Abstract registration', 'Paper submission', 'Round 2 abstract'",
                        },
                        "date_iso": {
                            "type": "string",
                            "description": "ISO 8601 datetime with timezone offset, e.g., 2026-07-15T23:59:00-07:00. Use AoE (UTC-12) if specified as 'Anywhere on Earth'.",
                        },
                        "timezone_note": {
                            "type": "string",
                            "description": "How the timezone was specified on the page (e.g., 'AoE', 'PST', 'UTC').",
                        },
                        "round": {
                            "type": ["string", "null"],
                            "description": "Round identifier if multi-round, otherwise null.",
                        },
                        "source_quote": {
                            "type": "string",
                            "description": "Exact text from the page that establishes this deadline (under 25 words).",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "0.0-1.0 confidence in this extraction.",
                        },
                    },
                    "required": ["name", "date_iso", "timezone_note", "round", "source_quote", "confidence"],
                },
            },
            "notes": {
                "type": "string",
                "description": "Anything notable: TBD deadlines, ambiguity, multi-track structure.",
            },
        },
        "required": ["is_cfp_page", "conference_year", "is_past", "deadlines", "notes"],
    },
}

LINK_RESOLUTION_SCHEMA = {
    "name": "link_resolution",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "current_cfp_url": {
                "type": ["string", "null"],
                "description": "Absolute URL of the page most likely to contain the current/upcoming CFP. Null if none found.",
            },
            "reasoning": {"type": "string"},
        },
        "required": ["current_cfp_url", "reasoning"],
    },
}


# ---------- Helpers ----------

@dataclass
class Conference:
    id: str
    name: str
    seed_url: str
    hints: str = ""


def load_conferences() -> list[Conference]:
    with open(CONFIG_FILE) as f:
        raw = yaml.safe_load(f)
    return [Conference(**c) for c in raw["conferences"]]


def load_previous() -> dict[str, Any]:
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text())
    return {"generated_at": None, "conferences": {}}


def fetch(url: str) -> tuple[str, str] | None:
    """Returns (final_url, html) or None on failure."""
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": "deadline-tracker (github.com/yourname/deadline-tracker)"},
        ) as c:
            r = c.get(url)
            r.raise_for_status()
            return str(r.url), r.text
    except Exception as e:
        print(f"  fetch failed for {url}: {e}", file=sys.stderr)
        return None


def extract_text(html: str, url: str) -> str:
    text = trafilatura.extract(html, url=url, include_links=True, include_tables=True)
    if not text:
        # Fallback to raw HTML truncated; the LLM can still parse it.
        text = html
    return text[:MAX_PAGE_CHARS]


def extract_links(html: str, base_url: str) -> list[dict[str, str]]:
    """Pull <a href> + link text. Used when resolving a hub page to a CFP page."""
    import re
    links = []
    seen = set()
    for m in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.IGNORECASE | re.DOTALL):
        href, text = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        absolute = urljoin(base_url, href)
        # Stay on a related domain (same registrable domain or subdomain).
        base_host = urlparse(base_url).hostname or ""
        link_host = urlparse(absolute).hostname or ""
        if not (link_host == base_host or link_host.endswith("." + ".".join(base_host.split(".")[-2:]))):
            continue
        key = (absolute, text[:80])
        if key in seen:
            continue
        seen.add(key)
        links.append({"url": absolute, "text": text[:120]})
        if len(links) >= 60:
            break
    return links


def llm_json(prompt: str, schema: dict, retries: int = 3) -> dict | None:
    """Call GitHub Models with structured output. Retries with backoff."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_schema", "json_schema": schema},
                temperature=0.0,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            wait = 2 ** attempt
            print(f"  LLM call failed (attempt {attempt+1}): {e}; retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    return None


def resolve_cfp_url(conf: Conference, seed_html: str, seed_url: str) -> str | None:
    """If the seed page isn't the CFP itself, find the right link."""
    links = extract_links(seed_html, seed_url)
    if not links:
        return None
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = f"""Today is {today}. I'm tracking deadlines for the {conf.name} conference.

The page at {seed_url} contains these links. Find the URL most likely to be the
current/upcoming call for papers (CFP) page with submission deadlines.

Prefer pages for the NEXT upcoming edition. If multiple years are present
(e.g., 2026 and 2027 sites), pick the one whose deadlines have not yet passed.
If the seed page already looks like a CFP, return null.

Hints from config: {conf.hints or "(none)"}

Links:
{json.dumps(links, indent=2)}
"""
    result = llm_json(prompt, LINK_RESOLUTION_SCHEMA)
    if result:
        return result.get("current_cfp_url")
    return None


def extract_deadlines(conf: Conference, page_text: str, page_url: str) -> dict | None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = f"""Today is {today}. Extract submission deadlines for {conf.name} from the page below.

Rules:
- Only include submission-related deadlines (abstract registration, paper submission,
  rebuttal, notification). Skip travel, registration, workshop, camera-ready unless
  they are the main submission milestones.
- For each deadline, return an exact ISO 8601 datetime with timezone offset.
  AoE = UTC-12. If no time is given, assume 23:59 local.
- Set is_past=true only if ALL deadlines on this page are already past.
- Confidence: 1.0 = explicit date with timezone; 0.5 = ambiguous or "TBD-ish";
  flag anything you're not sure about with lower confidence and explain in notes.

Hints from config: {conf.hints or "(none)"}

Page URL: {page_url}

--- PAGE CONTENT ---
{page_text}
--- END ---
"""
    return llm_json(prompt, DEADLINE_EXTRACTION_SCHEMA)


# ---------- Per-conference pipeline ----------

def process(conf: Conference, previous: dict) -> dict:
    print(f"\n=== {conf.name} ({conf.id}) ===")
    prev_entry = previous.get("conferences", {}).get(conf.id, {})
    result = {
        "id": conf.id,
        "name": conf.name,
        "seed_url": conf.seed_url,
        "resolved_url": None,
        "last_checked": datetime.now(timezone.utc).isoformat(),
        "last_success": prev_entry.get("last_success"),
        "stale_runs": prev_entry.get("stale_runs", 0),
        "status": "error",
        "deadlines": prev_entry.get("deadlines", []),
        "notes": "",
    }

    # 1. Try the cached resolved URL first if we have one and it's recent.
    candidate_url = prev_entry.get("resolved_url") or conf.seed_url
    fetched = fetch(candidate_url)
    if not fetched:
        # Fall back to seed if cached resolved URL is dead.
        if candidate_url != conf.seed_url:
            fetched = fetch(conf.seed_url)
            candidate_url = conf.seed_url
    if not fetched:
        result["stale_runs"] += 1
        result["notes"] = "Failed to fetch."
        return result
    final_url, html = fetched

    # 2. Try extraction directly. If the page isn't the CFP, resolve and retry.
    text = extract_text(html, final_url)
    extraction = extract_deadlines(conf, text, final_url)

    if extraction and not extraction["is_cfp_page"]:
        print("  not a CFP page, resolving...")
        resolved = resolve_cfp_url(conf, html, final_url)
        if resolved and resolved != final_url:
            fetched2 = fetch(resolved)
            if fetched2:
                final_url, html = fetched2
                text = extract_text(html, final_url)
                extraction = extract_deadlines(conf, text, final_url)

    # 3. If still past, try resolving from the seed (maybe a new year's site exists).
    if extraction and extraction["is_past"] and candidate_url != conf.seed_url:
        print("  all deadlines past, checking seed for newer site...")
        seed_fetch = fetch(conf.seed_url)
        if seed_fetch:
            seed_url, seed_html = seed_fetch
            resolved = resolve_cfp_url(conf, seed_html, seed_url)
            if resolved and resolved != final_url:
                fetched3 = fetch(resolved)
                if fetched3:
                    final_url, html = fetched3
                    text = extract_text(html, final_url)
                    extraction = extract_deadlines(conf, text, final_url)

    result["resolved_url"] = final_url

    if not extraction:
        result["stale_runs"] += 1
        result["notes"] = "LLM extraction failed."
        return result

    if not extraction["is_cfp_page"] or not extraction["deadlines"]:
        result["stale_runs"] += 1
        result["status"] = "no_deadlines"
        result["notes"] = extraction.get("notes") or "No deadlines found."
        return result

    result["status"] = "ok"
    result["stale_runs"] = 0
    result["last_success"] = result["last_checked"]
    result["conference_year"] = extraction.get("conference_year")
    result["deadlines"] = extraction["deadlines"]
    result["notes"] = extraction.get("notes", "")
    print(f"  ✓ {len(extraction['deadlines'])} deadline(s)")
    return result


# ---------- Main ----------

def main() -> int:
    confs = load_conferences()
    previous = load_previous()
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "conferences": {},
    }
    for conf in confs:
        try:
            output["conferences"][conf.id] = process(conf, previous)
        except Exception as e:
            print(f"  unexpected error for {conf.id}: {e}", file=sys.stderr)
            prev_entry = previous.get("conferences", {}).get(conf.id, {})
            output["conferences"][conf.id] = {
                **prev_entry,
                "last_checked": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "stale_runs": prev_entry.get("stale_runs", 0) + 1,
                "notes": f"Unexpected error: {e}",
            }
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nWrote {DATA_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
