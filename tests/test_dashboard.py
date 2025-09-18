#!/usr/bin/env python3
"""Quick manual tester for the admin dashboard endpoints."""

from __future__ import annotations

import json
from typing import Any

import requests

BASE_URL = "http://localhost:8080"


def pretty(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def main() -> None:
    print("=== Admin Dashboard Smoke Test ===")

    resp = requests.get(f"{BASE_URL}/admin/dashboard")
    print("/admin/dashboard:", resp.status_code)
    resp.raise_for_status()

    resp = requests.get(f"{BASE_URL}/admin/api/health")
    print("/admin/api/health:", resp.status_code)
    if resp.ok:
        health = resp.json()
        print("  status:", health.get("status"))
        print("  uptime:", health.get("uptime"))
        print("  total requests:", health.get("metrics", {}).get("total_requests"))

    resp = requests.get(f"{BASE_URL}/admin/api/metrics")
    print("/admin/api/metrics:", resp.status_code)
    if resp.ok:
        data = resp.json()
        print("  metrics keys:", list(data.get("summary", {}).keys()))

    resp = requests.get(f"{BASE_URL}/admin/api/sessions")
    print("/admin/api/sessions:", resp.status_code)
    if resp.ok:
        sessions = resp.json()
        print("  sessions returned:", sessions.get("total_count"))

    resp = requests.get(f"{BASE_URL}/admin/api/errors")
    print("/admin/api/errors:", resp.status_code)
    if resp.ok:
        errors = resp.json()
        print("  total errors (recent):", errors.get("total_errors"))

    print("\nDone.")


if __name__ == "__main__":
    main()
