#!/usr/bin/env python3
"""Live preflight before a recording take. Free calls only (model lists); never prints keys.

Checks:
  1. Meta Model API GET /v1/models via the egress proxy (default http://127.0.0.1:8791)
     and that sam-3.1 + muse-spark-1.3-contributor are listed (empty list => SAM 404s => stage 1 unknown).
  2. Baseten GET /v1/models reachability with BASETEN_API_KEY and that the Spark Open model is listed.
Exit code 0 only if everything is green.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

REQUIRED_META = ["sam-3.1", "muse-spark-1.3-contributor"]


def _ids(resp: httpx.Response) -> list[str]:
    try:
        data = resp.json()
    except Exception:
        return []
    items = data.get("data", data if isinstance(data, list) else [])
    return [str(m.get("id", m)) if isinstance(m, dict) else str(m) for m in items]


def check_meta() -> bool:
    key = os.getenv("META_API_KEY", "")
    base = os.getenv("META_BASE_URL", "https://api.meta.ai/v1").rstrip("/")
    proxy = os.getenv("REWIND_META_PROXY") or os.getenv("HTTPS_PROXY") or "http://127.0.0.1:8791"
    if not key:
        print("[META] FAIL  META_API_KEY not set")
        return False
    try:
        with httpx.Client(proxy=proxy, timeout=10.0, trust_env=False) as c:
            r = c.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError as exc:
        print(f"[META] FAIL  transport error via proxy {proxy}: {type(exc).__name__} (is the tunnel/computer online?)")
        return False
    ids = _ids(r)
    missing = [m for m in REQUIRED_META if m not in ids]
    ok = r.status_code == 200 and not missing
    print(f"[META] {'OK  ' if ok else 'FAIL'}  HTTP {r.status_code}, {len(ids)} models listed"
          + (f", missing {missing}" if missing else "") + f" (req {r.headers.get('x-request-id', '-')})")
    if r.status_code == 200 and not ids:
        print("       model list EMPTY: sam-3.1 calls will 404 and stage 1 will return unknown. Wait and retry.")
    return ok


def check_baseten() -> bool:
    key = os.getenv("BASETEN_API_KEY", "")
    base = os.getenv("BASETEN_BASE_URL", "https://inference.baseten.co/v1").rstrip("/")
    model = os.getenv("BASETEN_SPARK_MODEL", "meta-models/muse-spark-open-1026")
    if not key:
        print("[BASETEN] FAIL  BASETEN_API_KEY not set")
        return False
    try:
        with httpx.Client(timeout=10.0, trust_env=False) as c:
            r = c.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError as exc:
        print(f"[BASETEN] FAIL  transport error: {type(exc).__name__}")
        return False
    ids = _ids(r)
    ok = r.status_code == 200 and (model in ids or not ids)
    note = "" if model in ids else (" (model list not exposed; reachability only)" if not ids else f", {model} NOT listed")
    print(f"[BASETEN] {'OK  ' if ok else 'FAIL'}  HTTP {r.status_code}, {len(ids)} models{note}")
    return ok


if __name__ == "__main__":
    results = [check_meta(), check_baseten()]
    print("PREFLIGHT:", "GREEN - ready to record" if all(results) else "RED - do not record yet")
    sys.exit(0 if all(results) else 1)
