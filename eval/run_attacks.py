"""Run the 15 attack snippets in eval/attacks/ through the Docker sandbox.

    python eval/run_attacks.py

An attack is CONTAINED when its output never contains "ESCAPED" and the run ends
within the time limit. Snippets marked `expect_timeout` must be killed by the
timer. Afterwards we check (a) a marker file written by one run is gone in the
next run and (b) no sandbox containers are left behind.
Refuses to run with the local (unsafe) sandbox: a fork bomb there would hit the host.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistant.config import Settings, load_dotenv  # noqa: E402
from assistant.sandbox import DockerSandbox  # noqa: E402


def main() -> int:
    load_dotenv(ROOT / ".env")
    s = Settings()
    if s.sandbox_backend != "docker":
        print("Refusing: attacks must run against SANDBOX_BACKEND=docker.")
        return 2
    sbx = DockerSandbox(s.sandbox_image, s.sandbox_timeout_s, s.sandbox_memory_mb, s.sandbox_max_output)
    smoke = sbx.run("print('ok')")
    if smoke.stdout.strip() != "ok":
        print("Sandbox is not working, fix this first:", smoke.stderr or smoke)
        return 2

    limit_ms = (s.sandbox_timeout_s + DockerSandbox.STARTUP_GRACE_S) * 1000
    rows = []
    for f in sorted((ROOT / "eval" / "attacks").glob("*.py")):
        code = f.read_text(encoding="utf-8")
        r = sbx.run(code)
        expect_timeout = "expect_timeout" in code.splitlines()[0]
        escaped = "ESCAPED" in r.stdout
        finished = r.duration_ms <= limit_ms + 2000
        contained = (not escaped) and finished and (r.timed_out if expect_timeout else True)
        rows.append({"attack": f.stem, "contained": contained, "exit_code": r.exit_code,
                     "timed_out": r.timed_out, "ms": r.duration_ms,
                     "output": (r.stdout.strip() or r.stderr.strip())[-200:]})
        print(f"{'CONTAINED' if contained else '*** ESCAPED ***':<16} {f.stem:<24} exit={r.exit_code} "
              f"{r.duration_ms} ms  {rows[-1]['output'][:80]!r}")

    persist = sbx.run("import os; print(os.path.exists('/tmp/persist_marker'))")
    no_persist = persist.stdout.strip() == "False"
    leftover = subprocess.run(["docker", "ps", "-aq", "--filter", "name=prv201-sbx-"],
                              capture_output=True, text=True).stdout.split()
    print(f"\nState persists between runs: {'no (good)' if no_persist else 'YES (bad)'}")
    print(f"Leftover sandbox containers: {len(leftover)}")

    n_ok = sum(r["contained"] for r in rows)
    print(f"\n{n_ok}/{len(rows)} attacks contained, {len(rows) - n_ok} escapes.")
    out = ROOT / "eval" / "results" / f"attacks_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"image": s.sandbox_image, "contained": n_ok, "total": len(rows),
                               "no_persistence": no_persist, "leftover_containers": len(leftover),
                               "results": rows}, indent=2), encoding="utf-8")
    print("Saved", out.relative_to(ROOT))
    return 0 if n_ok == len(rows) and no_persist and not leftover else 1


if __name__ == "__main__":
    sys.exit(main())
