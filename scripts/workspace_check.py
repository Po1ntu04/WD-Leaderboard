from __future__ import annotations

import argparse
import json
import socket
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.35)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the local WD-Leaderboard workspace shape.")
    parser.add_argument("--results-dir", default="platform/results")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    results_dir = ROOT / args.results_dir
    dashboard_py = ROOT / "platform" / "app" / "dashboard.py"
    dashboard_text = dashboard_py.read_text(encoding="utf-8") if dashboard_py.exists() else ""
    grouped_tabs = all(label in dashboard_text for label in ["Overview", "Leaderboard", "Diagnostics", "Gold Review", "Experimental"])
    old_tabs = "P0-1 Overview Dashboard" in dashboard_text or "P1-6 Student" in dashboard_text

    manifest_path = results_dir / "long_tables_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    print(f"Workspace: {ROOT}")
    print(f"Git HEAD:  {run(['git', 'rev-parse', '--short', 'HEAD']) or 'unknown'}")
    branch = run(["git", "branch", "--show-current"]) or run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    print(f"Branch:    {'detached HEAD' if branch == 'HEAD' else branch}")
    print(f"Dirty:     {'yes' if run(['git', 'status', '--short']) else 'no'}")
    print(f"Dashboard: {'new grouped tabs' if grouped_tabs and not old_tabs else 'old/unknown tab layout'}")
    print(f"Results:   {results_dir} ({'exists' if results_dir.exists() else 'missing'})")
    print(f"Port {args.port}: {'in use' if port_open(args.port) else 'free'}")
    if manifest:
        tables = manifest.get("tables", manifest)
        print("Long tables manifest: present")
        if isinstance(tables, dict):
            for name in ["sentence_score_table", "boundary_table", "span_error_table"]:
                item = tables.get(name, {}) if isinstance(tables.get(name, {}), dict) else {}
                rows = item.get("row_count", item.get("rows", "?"))
                print(f"  - {name}: rows={rows}")
    else:
        print("Long tables manifest: missing or unreadable")


if __name__ == "__main__":
    main()
