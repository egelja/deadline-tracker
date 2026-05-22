# Conference Deadline Tracker

Self-updating deadline tracker for academic conferences. Runs nightly on GitHub Actions, extracts deadlines using GitHub Models (free LLM tier), and publishes a static site on GitHub Pages.

## How it works

1. `conferences.yaml` lists conferences with a stable seed URL each.
2. Nightly, `scripts/extract.py` fetches each seed page. If it's a hub (not a CFP), it asks the LLM to find the current CFP link, then fetches that. Cached resolved URLs are reused until they go stale.
3. The page is stripped to clean text and the LLM extracts deadlines as structured JSON (name, ISO date, timezone, source quote, confidence).
4. `scripts/diff.py` compares the new state with the prior commit and writes human-readable changes to `CHANGELOG.md`.
5. Astro builds a static site from `data/deadlines.json`, deployed to GitHub Pages.

## Setup

1. Fork or create from this repo (public).
2. Enable GitHub Pages: Settings → Pages → Source: GitHub Actions.
3. Enable Models access (free): Settings → Code, planning, and automation → Actions → General → Workflow permissions includes `models: read` (already set in workflow).
4. Edit `conferences.yaml` to add conferences you care about.
5. Push. The workflow runs on push, on a nightly schedule, and via manual dispatch.

## Adding a conference

```yaml
- id: short-slug          # used as the JSON key, keep stable
  name: Display Name
  seed_url: https://...   # stable URL; can be a hub page
  hints: "Optional free-text hints passed to the LLM"
```

## When it breaks

- A conference's `stale_runs` counter climbs and the site shows ⚠ next to its row.
- Low-confidence extractions show `?` next to the row.
- Check the action logs, then the `source_quote` field in `data/deadlines.json` to see what the LLM read.
- If the seed URL is dead, update it. If the page format is weird, add a `hints:` line.

## Cost

GitHub Models free tier: 150 GPT-4o-mini requests/day. With ~30 conferences × 2 calls each, you use ~60/day. Fine.
