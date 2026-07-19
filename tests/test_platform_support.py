"""Mac (Apple Silicon) port: unit tests for the OS-detection seam module.

The dev machine is Windows, so the macOS/Linux branches are verified by
patching otoweave_app.platform_support.IS_WINDOWS/IS_MACOS/IS_LINUX
rather than by actually running on those platforms. Windows-path behaviour
must stay byte-for-byte identical to before the Mac port, so several tests
assert the real (unpatched, Windows) behaviour too.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from otoweave_app import platform_support as ps


def _force_os(windows: bool, macos: bool, linux: bool):
    """Context manager patching the three OS flags together."""
    return patch.multiple(
        ps,
        IS_WINDOWS=windows,
        IS_MACOS=macos,
        IS_LINUX=linux,
    )


class ExecutableNameTests(unittest.TestCase):
    def test_windows_appends_exe(self) -> None:
        with _force_os(True, False, False):
            self.assertEqual(ps.executable_name("ffmpeg"), "ffmpeg.exe")
            self.assertEqual(ps.executable_name("llama-server"), "llama-server.exe")

    def test_macos_keeps_bare_name(self) -> None:
        with _force_os(False, True, False):
            self.assertEqual(ps.executable_name("ffmpeg"), "ffmpeg")
            self.assertEqual(ps.executable_name("llama-server"), "llama-server")

    def test_linux_keeps_bare_name(self) -> None:
        with _force_os(False, False, True):
            self.assertEqual(ps.executable_name("ffmpeg"), "ffmpeg")

    def test_real_platform_is_windows_in_this_dev_environment(self) -> None:
        # Sanity check that the module's own OS detection matches this CI
        # box, so the "real Windows" assertions elsewhere are meaningful.
        self.assertTrue(ps.IS_WINDOWS)
        self.assertFalse(ps.IS_MACOS)


class ResolveFfmpegTests(unittest.TestCase):
    def test_bundled_binary_is_preferred_on_any_os(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundled = root / "engines" / "ffmpeg" / "ffmpeg.exe"
            bundled.parent.mkdir(parents=True)
            bundled.write_bytes(b"stub")
            with _force_os(True, False, False):
                self.assertEqual(ps.resolve_ffmpeg(root), bundled)

    def test_windows_without_bundled_binary_returns_expected_path_unchecked(self) -> None:
        # Windows never falls back to PATH: the missing bundled path is
        # returned so callers keep their existing "file missing" error.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with _force_os(True, False, False):
                result = ps.resolve_ffmpeg(root)
            self.assertEqual(result, root / "engines" / "ffmpeg" / "ffmpeg.exe")

    def test_macos_falls_back_to_path_when_bundled_binary_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with _force_os(False, True, False), patch(
                "shutil.which", return_value="/opt/homebrew/bin/ffmpeg"
            ):
                result = ps.resolve_ffmpeg(root)
            self.assertEqual(result, Path("/opt/homebrew/bin/ffmpeg"))

    def test_macos_without_path_ffmpeg_returns_bundled_path_unchecked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with _force_os(False, True, False), patch("shutil.which", return_value=None):
                result = ps.resolve_ffmpeg(root)
            self.assertEqual(result, root / "engines" / "ffmpeg" / "ffmpeg")

    def test_real_windows_bundled_engine_path_matches_repo_layout(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        self.assertEqual(
            ps.resolve_ffmpeg(project_root),
            project_root / "engines" / "ffmpeg" / "ffmpeg.exe",
        )


class TotalPhysicalRamBytesTests(unittest.TestCase):
    def test_dispatches_to_windows_helper(self) -> None:
        with _force_os(True, False, False), patch.object(
            ps, "_windows_ram_bytes", return_value=123
        ) as windows_helper, patch.object(
            ps, "_sysctl_ram_bytes"
        ) as mac_helper, patch.object(
            ps, "_linux_ram_bytes"
        ) as linux_helper:
            self.assertEqual(ps.total_physical_ram_bytes(), 123)
            windows_helper.assert_called_once()
            mac_helper.assert_not_called()
            linux_helper.assert_not_called()

    def test_dispatches_to_macos_helper(self) -> None:
        with _force_os(False, True, False), patch.object(
            ps, "_sysctl_ram_bytes", return_value=8 * 1024**3
        ) as mac_helper, patch.object(
            ps, "_windows_ram_bytes"
        ) as windows_helper:
            self.assertEqual(ps.total_physical_ram_bytes(), 8 * 1024**3)
            mac_helper.assert_called_once()
            windows_helper.assert_not_called()

    def test_dispatches_to_linux_helper(self) -> None:
        with _force_os(False, False, True), patch.object(
            ps, "_linux_ram_bytes", return_value=16 * 1024**3
        ) as linux_helper:
            self.assertEqual(ps.total_physical_ram_bytes(), 16 * 1024**3)
            linux_helper.assert_called_once()

    def test_sysctl_helper_parses_hw_memsize_output(self) -> None:
        fake_result = type(
            "Result", (), {"returncode": 0, "stdout": "8589934592\n"}
        )()
        with patch("subprocess.run", return_value=fake_result):
            self.assertEqual(ps._sysctl_ram_bytes(), 8589934592)

    def test_sysctl_helper_returns_zero_on_failure(self) -> None:
        with patch("subprocess.run", side_effect=OSError("no sysctl")):
            self.assertEqual(ps._sysctl_ram_bytes(), 0)

    def test_linux_helper_parses_proc_meminfo(self) -> None:
        meminfo = "MemTotal:       16384000 kB\nMemFree:         1000000 kB\n"
        with patch("builtins.open", mock_open(read_data=meminfo)):
            self.assertEqual(ps._linux_ram_bytes(), 16384000 * 1024)

    def test_linux_helper_returns_zero_when_meminfo_missing(self) -> None:
        with patch("builtins.open", side_effect=OSError("missing")):
            self.assertEqual(ps._linux_ram_bytes(), 0)

    def test_real_windows_ram_is_positive_on_this_machine(self) -> None:
        # Exercises the real (unpatched) Windows code path end to end.
        self.assertGreater(ps.total_physical_ram_bytes(), 0)


class ChildProcessSupervisionTests(unittest.TestCase):
    """The orphan-process safety net: Windows Job Object vs POSIX process
    group. See windows_job.py / posix_process.py for the primitives."""

    def test_windows_create_job_delegates_to_windows_job_module(self) -> None:
        with _force_os(True, False, False), patch(
            "otoweave_app.windows_job.create_kill_on_close_job",
            return_value=999,
        ) as create:
            self.assertEqual(ps.create_kill_on_close_job(), 999)
            create.assert_called_once()

    def test_posix_create_job_returns_none_without_calling_windows_job(self) -> None:
        with _force_os(False, True, False), patch(
            "otoweave_app.windows_job.create_kill_on_close_job"
        ) as create:
            self.assertIsNone(ps.create_kill_on_close_job())
            create.assert_not_called()

    def test_windows_popen_kwargs_are_empty(self) -> None:
        with _force_os(True, False, False):
            self.assertEqual(ps.child_popen_kwargs(), {})

    def test_posix_popen_kwargs_start_a_new_session(self) -> None:
        with _force_os(False, True, False):
            self.assertEqual(ps.child_popen_kwargs(), {"start_new_session": True})
        with _force_os(False, False, True):
            self.assertEqual(ps.child_popen_kwargs(), {"start_new_session": True})

    def test_windows_assign_process_to_job_delegates(self) -> None:
        sentinel_process = object()
        with _force_os(True, False, False), patch(
            "otoweave_app.windows_job.assign_process_to_job",
            return_value=True,
        ) as assign:
            self.assertTrue(ps.assign_process_to_job(42, sentinel_process))
            assign.assert_called_once_with(42, sentinel_process)

    def test_posix_assign_process_to_job_is_a_noop(self) -> None:
        sentinel_process = object()
        with _force_os(False, True, False), patch(
            "otoweave_app.windows_job.assign_process_to_job"
        ) as assign:
            self.assertFalse(ps.assign_process_to_job(None, sentinel_process))
            assign.assert_not_called()

    def test_windows_terminate_calls_kill_when_still_running(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.killed = False

            def poll(self):
                return None if not self.killed else 1

            def kill(self) -> None:
                self.killed = True

        process = FakeProcess()
        with _force_os(True, False, False):
            ps.terminate_child_process(process)
        self.assertTrue(process.killed)

    def test_posix_terminate_uses_process_group_kill(self) -> None:
        sentinel_process = object()
        with _force_os(False, True, False), patch(
            "otoweave_app.posix_process.kill_process_group"
        ) as kill_group:
            ps.terminate_child_process(sentinel_process)
            kill_group.assert_called_once_with(sentinel_process)


if __name__ == "__main__":
    unittest.main()
