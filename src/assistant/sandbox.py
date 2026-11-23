"""run_python sandbox.

DockerSandbox is the only back end allowed in normal use. Each call starts a
fresh container that is destroyed afterwards (--rm) and has:
  no network, read-only root filesystem, non-root user (nobody), all Linux
  capabilities dropped, no privilege escalation, 512 MB memory (no swap),
  1 CPU, 64 processes max, a 16 MB noexec /tmp, and a 10 s kill timer.

LocalSubprocessSandbox exists only so unit tests can run on machines without
Docker. It is NOT a security boundary and refuses to start unless
ALLOW_UNSAFE_LOCAL_SANDBOX=1 is set.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Protocol

from .types import SandboxResult


class Sandbox(Protocol):
    def run(self, code: str) -> SandboxResult: ...


def _trim(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f"\n...[truncated {len(s) - n} chars]"


class DockerSandbox:
    STARTUP_GRACE_S = 5  # container start-up time on top of the code's own limit

    def __init__(self, image: str, timeout_s: int = 10, memory_mb: int = 512, max_output: int = 4000):
        self.image, self.timeout_s, self.memory_mb, self.max_output = image, timeout_s, memory_mb, max_output

    def command(self, name: str) -> list[str]:
        return [
            "docker", "run", "--rm", "-i", "--name", name,
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            "--memory", f"{self.memory_mb}m", "--memory-swap", f"{self.memory_mb}m",
            "--cpus", "1", "--pids-limit", "64",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534",
            "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp", "-e", "MPLBACKEND=Agg",
            "-w", "/tmp",
            self.image,
            # inner kill timer: counts only code time, not container start-up
            "timeout", "-s", "KILL", str(self.timeout_s), "python", "-I", "-",
        ]

    def run(self, code: str) -> SandboxResult:
        name = f"prv201-sbx-{uuid.uuid4().hex[:12]}"
        t0 = time.monotonic()
        try:
            p = subprocess.run(self.command(name), input=code, capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=self.timeout_s + self.STARTUP_GRACE_S)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            return SandboxResult(False, None, "", "", True, int((time.monotonic() - t0) * 1000))
        except FileNotFoundError as e:
            return SandboxResult(False, None, "", f"Docker is not installed: {e}", False, 0)
        ms = int((time.monotonic() - t0) * 1000)
        if p.returncode == 125:  # docker itself failed (image missing, daemon down)
            return SandboxResult(False, None, "", f"Sandbox failed to start: {p.stderr.strip()}", False, ms)
        stderr = p.stderr
        # 137 = SIGKILL: either our `timeout` fired or the kernel OOM-killed the process.
        timed_out = p.returncode == 137 and ms >= self.timeout_s * 1000
        if p.returncode == 137 and not timed_out:
            stderr += f"\nKilled: process exceeded the {self.memory_mb} MB memory limit."
        return SandboxResult(ok=p.returncode == 0, exit_code=p.returncode,
                             stdout=_trim(p.stdout, self.max_output), stderr=_trim(stderr, self.max_output),
                             timed_out=timed_out, duration_ms=ms)


class LocalSubprocessSandbox:
    """UNSAFE. Development/test runner with basic rlimits only."""

    def __init__(self, timeout_s: int = 10, memory_mb: int = 512, max_output: int = 4000):
        if os.environ.get("ALLOW_UNSAFE_LOCAL_SANDBOX") != "1":
            raise RuntimeError("LocalSubprocessSandbox is not isolated. Set ALLOW_UNSAFE_LOCAL_SANDBOX=1 "
                               "only for tests; use SANDBOX_BACKEND=docker otherwise.")
        self.timeout_s, self.memory_mb, self.max_output = timeout_s, memory_mb, max_output

    def _limits(self):  # runs in the child before exec (POSIX only; skipped on Windows)
        import resource

        mem = self.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        # CPU cap 1 s above the wall timeout so the wall timer normally fires first
        resource.setrlimit(resource.RLIMIT_CPU, (self.timeout_s + 1, self.timeout_s + 1))

    def run(self, code: str) -> SandboxResult:
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory() as tmp:
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tmp, "MPLBACKEND": "Agg",
                   "MPLCONFIGDIR": tmp, "PYTHONIOENCODING": "utf-8"}
            if os.name == "nt":  # Windows: Python can't start without SYSTEMROOT; no rlimits available
                env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
            extra = {"preexec_fn": self._limits} if os.name == "posix" else {}
            try:
                p = subprocess.run([sys.executable, "-I", "-"], input=code, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace",
                                   cwd=tmp, env=env, timeout=self.timeout_s, **extra)
            except subprocess.TimeoutExpired:
                return SandboxResult(False, None, "", "", True, int((time.monotonic() - t0) * 1000))
        import signal

        cpu_killed = os.name == "posix" and p.returncode in (-24, -9)  # CPU rlimit hit (SIGXCPU/SIGKILL)
        return SandboxResult(p.returncode == 0, p.returncode, _trim(p.stdout, self.max_output),
                             _trim(p.stderr, self.max_output), cpu_killed, int((time.monotonic() - t0) * 1000))


def build_sandbox(settings) -> Sandbox:
    if settings.sandbox_backend == "local":
        return LocalSubprocessSandbox(settings.sandbox_timeout_s, settings.sandbox_memory_mb, settings.sandbox_max_output)
    return DockerSandbox(settings.sandbox_image, settings.sandbox_timeout_s, settings.sandbox_memory_mb,
                         settings.sandbox_max_output)
