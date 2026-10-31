"""Smoke test. Usage:
  python scripts/smoke_test.py <base_url>   # checks /_stcore/health on a running app
  python scripts/smoke_test.py --local      # calls ask() in-process (needs ANTHROPIC_API_KEY)
Both may be combined. Makes real LLM calls in --local mode.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLEAN_Q = "What PPE do I need for sanitizer changeover?"
CONFLICT_Q = "A bottle fell over on conveyor C2. Can I clear it with the conveyor in jog mode?"


def check_health(base_url: str) -> None:
    url = base_url.rstrip("/") + "/_stcore/health"
    with urllib.request.urlopen(url, timeout=15) as r:
        assert r.status == 200, f"{url} returned {r.status}"
    print(f"ok  health: {url}")


def check_local() -> None:
    from dotenv import load_dotenv

    import config

    load_dotenv(config.ROOT / ".env")
    from app.pipeline import ask

    clean = ask(CLEAN_Q, write_log=False)
    assert "safety" in clean.searched_domains, f"clean question searched {clean.searched_domains}"
    assert not clean.not_covered and clean.answer.strip(), "clean question got no answer"
    print(f"ok  clean question: searched {clean.searched_domains}")

    conflict = ask(CONFLICT_Q, write_log=False)
    assert {"safety", "maintenance"} <= set(conflict.searched_domains), conflict.searched_domains
    assert conflict.conflicts, "expected a conflict for the jog-mode question"
    print(f"ok  conflict question: {len(conflict.conflicts)} conflict(s), "
          f"escalate_to={conflict.conflicts[0].escalate_to}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url", nargs="?")
    ap.add_argument("--local", action="store_true")
    args = ap.parse_args()
    if not args.base_url and not args.local:
        ap.error("give a base_url and/or --local")
    try:
        if args.base_url:
            check_health(args.base_url)
        if args.local:
            check_local()
    except (AssertionError, OSError) as e:
        print(f"FAIL {e}", file=sys.stderr)
        return 1
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
