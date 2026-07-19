"""Stream subprocess stdout/stderr live in Jupyter (no buffering)."""

from __future__ import annotations

import os
import subprocess
import sys


def run_live(cmd: list[str], *, cwd: str | None = None) -> None:
    """Run command and print each output line immediately."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    argv = list(cmd)
    if argv and argv[0] in ("python", "python3"):
        argv = [argv[0], "-u", *argv[1:]]

    process = subprocess.Popen(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    if process.stdout is None:
        raise RuntimeError("stdout pipe not available")

    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, argv)
