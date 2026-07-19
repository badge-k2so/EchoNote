"""POSIX process-group helpers: the non-Windows counterpart to
windows_job.py's kill-on-close Job Object.

macOS/Linux have no direct equivalent of a Windows Job Object that
auto-kills children when the parent process closes. What POSIX does offer
is process groups: starting a child in its own session (start_new_session
at Popen time) means a later os.killpg() takes down that child and any
grandchildren it spawned in one call, instead of leaking them as orphans
when only the immediate child is killed.

These are plain OS-level primitives with no platform branching of their
own; otoweave_app.platform_support is the seam callers should use
(create_kill_on_close_job / child_popen_kwargs / assign_process_to_job /
terminate_child_process) so call sites do not need to import this module
directly or special-case the OS themselves.
"""
from __future__ import annotations

import os
import signal
import subprocess


def new_session_popen_kwargs() -> dict:
    """Extra subprocess.Popen(...) kwargs so the child starts its own
    process group/session."""
    return {"start_new_session": True}


def kill_process_group(process: "subprocess.Popen") -> None:
    """Terminate the process and everything in its process group.

    Falls back to killing just the process when the group cannot be
    resolved (already exited, permission denied, or it was never started
    with its own session) so this is always a safe best-effort call."""
    try:
        if process.poll() is not None:
            return
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGTERM)
        return
    except Exception:
        pass
    try:
        if process.poll() is None:
            process.kill()
    except OSError:
        pass
