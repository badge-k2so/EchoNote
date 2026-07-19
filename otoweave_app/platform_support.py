"""Cross-platform helpers so one codebase runs on Windows and Apple Silicon.

OtoWeave was built for GIGA-Windows laptops. The Mac (Apple Silicon) port
keeps a single source tree and branches at the few OS-specific seams:
audio recording, TTS, the ffmpeg/engine binaries, RAM detection, child
process supervision, and (in otoweave_app.py, owned separately) the
native file-open dialog. This module centralises the detection and the
small platform utilities those seams need.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import subprocess

IS_WINDOWS = sys.platform.startswith("win")
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")


def executable_name(base: str) -> str:
    """Append .exe on Windows only (e.g. ffmpeg / llama-server)."""
    return f"{base}.exe" if IS_WINDOWS else base


def resolve_ffmpeg(project_root: Path) -> Path:
    """Locate ffmpeg.

    Prefer the bundled arm64/x64 binary under engines/ffmpeg/, then fall
    back to an ffmpeg found on PATH (Homebrew installs /opt/homebrew/bin).
    The Path is returned even if nothing exists yet so callers keep their
    existing "file missing" error behaviour.
    """
    project_root = Path(project_root)
    bundled = project_root / "engines" / "ffmpeg" / executable_name("ffmpeg")
    if bundled.is_file():
        return bundled
    if not IS_WINDOWS:
        from shutil import which

        found = which("ffmpeg")
        if found:
            return Path(found)
    return bundled


def total_physical_ram_bytes() -> int:
    """Total physical RAM in bytes, or 0 when it cannot be determined.

    Used to pick the LLM context/batch profile — an 8 GB Apple Silicon
    Mac must get the low-memory profile just like a 4-8 GB GIGA laptop.
    """
    if IS_WINDOWS:
        return _windows_ram_bytes()
    if IS_MACOS:
        return _sysctl_ram_bytes()
    return _linux_ram_bytes()


def _windows_ram_bytes() -> int:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except Exception:
        pass
    return 0


def _sysctl_ram_bytes() -> int:
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return 0


def _linux_ram_bytes() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------
# Child-process supervision (the orphan-process safety net).
#
# Windows uses a kill-on-close Job Object (windows_job.py): if OtoWeave is
# force-closed, the OS also terminates any llama-server / summarize-script
# child still attached to the job, so it cannot outlive the app as a
# CPU/RAM-hogging orphan. POSIX has no exact equivalent, but starting each
# child in its own session (posix_process.py) lets a controlled shutdown
# kill the whole process group -- including any grandchildren the child
# spawns -- with one os.killpg() call. The four functions below are the
# single OS-branching seam callers (asr.py, llm_session.py, llm_chat.py)
# use instead of importing windows_job/posix_process directly.
# ---------------------------------------------------------------------


def create_kill_on_close_job() -> "int | None":
    """Windows: a kill-on-close Job Object handle. POSIX: None -- there is
    nothing to pre-create; each child gets its own session at Popen time
    via child_popen_kwargs()."""
    if IS_WINDOWS:
        from .windows_job import create_kill_on_close_job as _create

        return _create()
    return None


def child_popen_kwargs() -> dict:
    """Extra kwargs to merge into a supervised subprocess.Popen(...) call.

    Empty on Windows (the Job Object handles cleanup instead). On POSIX,
    starts the child in its own session so terminate_child_process() can
    later kill the whole process group."""
    if IS_WINDOWS:
        return {}
    from .posix_process import new_session_popen_kwargs

    return new_session_popen_kwargs()


def assign_process_to_job(job: "int | None", process: "subprocess.Popen") -> bool:
    """Attach a freshly started child to the kill-on-close job.

    Windows: registers it with the Job Object. POSIX: no-op -- the child
    already established its own session at Popen time (child_popen_kwargs())."""
    if IS_WINDOWS:
        from .windows_job import assign_process_to_job as _assign

        return _assign(job, process)
    return False


def terminate_child_process(process: "subprocess.Popen") -> None:
    """Force-kill a supervised child process.

    Windows: process.kill(). POSIX: kill the whole process group so
    grandchildren (if the child spawned any) cannot survive as orphans."""
    if IS_WINDOWS:
        try:
            if process.poll() is None:
                process.kill()
        except OSError:
            pass
        return
    from .posix_process import kill_process_group

    kill_process_group(process)
