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


def _html_to_text(html: str) -> str:
    """Strip scripts/styles/markup and return visible text. Fallback when
    trafilatura yields nothing usable."""
    import re
    # Drop script/style/noscript blocks entirely.
    html = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", html,
                  flags=re.IGNORECASE | re.DOTALL)
    # Drop head section (metadata, not visible content).
    html = re.sub(r"<head[^>]*>.*?</head>", " ", html,
                  flags=re.IGNORECASE | re.DOTALL)
    # Convert breaks/rows to newlines so dates on separate lines stay separate.
    html = re.sub(r"<(br|/p|/div|/li|/tr|/h[1-6])[^>]*>", "\n", html,
                  flags=re.IGNORECASE)
    # Strip remaining tags.
    text = re.sub(r"<[^>]+>", " ", html)
    # Unescape common entities.
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'")
                .replace("&quot;", '"').replace("&ndash;", "-").replace("&mdash;", "-"))
    # Collapse whitespace but preserve line structure.
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(ln for ln in lines if ln)
    return text


# A page with real CFP content should comfortably exceed this once cleaned.
MIN_CONTENT_CHARS = 200


def extract_text(html: str, url: str) -> str:
    text = trafilatura.extract(html, url=url, include_links=True, include_tables=True)
    if not text or len(text.strip()) < MIN_CONTENT_CHARS:
        # trafilatura failed or returned almost nothing — use our own cleaner
        # rather than dumping raw HTML (which is mostly boilerplate up front).
        fallback = _html_to_text(html)
        # Prefer whichever has more real content.
        if not text or len(fallback) > len(text):
            text = fallback
    return (text or "")[:MAX_PAGE_CHARS]


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
    """Given a page's links, pick the one most likely to lead to the current
    CFP. Handles both hub pages (pick the right year) and landing pages
    (follow the 'Call for Papers' link)."""
    links = extract_links(seed_html, seed_url)
    if not links:
        return None
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    year = datetime.now(timezone.utc).year
    prompt = f"""Today is {today}. I'm tracking submission deadlines for {conf.name}.

I'm on the page {seed_url} and need to find the link that leads to the page with
the actual submission DEADLINES for the CURRENT or NEXT upcoming edition.

How to choose:
- If you see an explicit "Call for Papers" / "CFP" / "Submissions" / "Important
  Dates" link, prefer it — that's where deadlines live, not the landing page.
- If links point to different years/editions (e.g. 2026 vs 2027), pick the
  EDITION WHOSE DEADLINES ARE STILL UPCOMING as of {today}. A conference held in
  year Y typically has its submission deadlines ~9-15 months earlier. So in
  {year}, the edition actively accepting submissions is often the {year+1}
  edition, NOT the {year} one (whose deadlines have likely passed). Prefer the
  LATER edition when in doubt, unless its deadlines are clearly not yet announced.
- Return the single best URL to navigate to next. If this page already clearly
  shows multiple dated submission deadlines, return null.

Hints from config: {conf.hints or "(none)"}

Links on this page:
{json.dumps(links, indent=2)}
"""
    result = llm_json(prompt, LINK_RESOLUTION_SCHEMA)
    if result:
        url = result.get("current_cfp_url")
        if url:
            print(f"  resolver: {url}  ({result.get('reasoning', '')[:80]})")
        return url
    return None


def extract_deadlines(conf: Conference, page_text: str, page_url: str) -> dict | None:
    # Guard: if we have almost no content, don't ask the LLM — it will hallucinate.
    if len(page_text.strip()) < MIN_CONTENT_CHARS:
        print(f"  content too thin ({len(page_text.strip())} chars), skipping extraction")
        return {
            "is_cfp_page": False,
            "conference_year": None,
            "is_past": False,
            "deadlines": [],
            "notes": "Page content too thin to extract (likely JS-rendered or fetch blocked).",
        }
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = f"""Today is {today}. Extract submission deadlines for {conf.name} STRICTLY from the page content below.

CRITICAL: Only use information present in the PAGE CONTENT. Do NOT use prior
knowledge about this conference. If the page does not contain dated deadlines,
set is_cfp_page=false and return an empty deadlines list. Never invent or guess
a date. Every `source_quote` MUST be copied verbatim from the PAGE CONTENT.

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
    result = llm_json(prompt, DEADLINE_EXTRACTION_SCHEMA)
    if not result:
        return None

    # Verify each quote actually appears in the source. Drop hallucinations.
    def normalize(s: str) -> str:
        return " ".join(s.lower().split())

    haystack = normalize(page_text)
    verified = []
    dropped = 0
    for d in result.get("deadlines", []):
        quote = normalize(d.get("source_quote", ""))
        # Require a non-trivial quote that is actually present in the page.
        if len(quote) >= 8 and quote in haystack:
            verified.append(d)
        else:
            dropped += 1
    if dropped:
        print(f"  dropped {dropped} deadline(s) whose quote was not found in page text")
        note = result.get("notes", "")
        result["notes"] = (note + f" [{dropped} unverified deadline(s) dropped]").strip()
    result["deadlines"] = verified
    if not verified:
        result["is_cfp_page"] = result.get("is_cfp_page", False) and False
    return result


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

    # 2. Iteratively crawl toward the real CFP page. Each hop: extract; if it's
    #    not a CFP with upcoming deadlines, ask the resolver for the next link.
    #    This handles multi-hop sites (sosp.org -> /2026/ -> /2026/cfp.html) and
    #    lets the resolver correct a wrong-year guess.
    MAX_HOPS = 4
    visited = {final_url}
    text = extract_text(html, final_url)
    extraction = extract_deadlines(conf, text, final_url)

    for hop in range(MAX_HOPS):
        good = (
            extraction
            and extraction.get("is_cfp_page")
            and extraction.get("deadlines")
            and not extraction.get("is_past")
        )
        if good:
            break

        # Decide where to look next.
        reason = (
            "not a CFP page" if (extraction and not extraction.get("is_cfp_page"))
            else "no upcoming deadlines" if (extraction and extraction.get("is_past"))
            else "no deadlines found"
        )
        print(f"  hop {hop}: {reason}, resolving next link...")
        resolved = resolve_cfp_url(conf, html, final_url)

        # If the current page yielded nothing and resolver gives nothing, try the
        # seed once (a newer edition's site may be linked there).
        if (not resolved or resolved in visited) and final_url != conf.seed_url:
            seed_fetch = fetch(conf.seed_url)
            if seed_fetch:
                s_url, s_html = seed_fetch
                resolved = resolve_cfp_url(conf, s_html, s_url)

        if not resolved or resolved in visited:
            break

        nxt = fetch(resolved)
        if not nxt:
            break
        final_url, html = nxt
        visited.add(final_url)
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
    # Debug mode: `python scripts/extract.py --debug osdi`
    # Fetches one conference, prints what text the LLM would actually see.
    # Use this to confirm the page is being read before trusting extractions.
    if len(sys.argv) >= 3 and sys.argv[1] == "--debug":
        target = sys.argv[2]
        conf = next((c for c in load_conferences() if c.id == target), None)
        if not conf:
            print(f"No conference with id '{target}'")
            sys.exit(1)
        fetched = fetch(conf.seed_url)
        if not fetched:
            print("FETCH FAILED")
            sys.exit(1)
        final_url, html = fetched
        text = extract_text(html, final_url)
        print(f"URL: {final_url}")
        print(f"HTML bytes: {len(html)}")
        print(f"Extracted text chars: {len(text)}")
        print("=" * 60)
        print(text[:3000])
        print("=" * 60)
        print(f"(showing first 3000 of {len(text)} chars)")
        sys.exit(0)
    sys.exit(main())
