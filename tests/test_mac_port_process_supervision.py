"""Mac (Apple Silicon) port: Metal GPU-layer wiring and the POSIX
orphan-process safety net (the non-Windows counterpart to windows_job.py's
kill-on-close Job Object).

The dev machine is Windows, so the macOS branches are exercised by
patching otoweave_app.platform_support.IS_MACOS / IS_WINDOWS.
"""
from __future__ import annotations

import queue as queue_module
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from otoweave_app.asr import Qwen3AsrRecognizer
from otoweave_app.llm_chat import (
    run_summarize_subprocess,
    run_template_summarize_subprocess,
)
from otoweave_app.llm_session import LlmSession
from otoweave_app import posix_process


# ---------------------------------------------------------------------
# Metal (--n_gpu_layers) wiring for the two summarize subprocess launchers.
# ---------------------------------------------------------------------


class SummarizeSubprocessMetalArgTests(unittest.TestCase):
    def _run_legacy(self, folder: Path) -> list[str]:
        captured: dict[str, list[str]] = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            from otoweave_app.llm_chat import _SummaryProcessResult

            return _SummaryProcessResult(0, "", "")

        with patch(
            "otoweave_app.llm_chat._run_summary_process",
            side_effect=fake_run,
        ):
            run_summarize_subprocess(
                SimpleNamespace(segments=[]),
                folder,
                Path("C:/fake/project"),
                Path("Qwen3.5-4B-Q4_K_M.gguf"),
            )
        return captured["command"]

    def _run_template(self, folder: Path) -> list[str]:
        template = {"id": "meeting_record", "name": "面談記録"}
        summaries_dir = folder / "postprocess" / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        (summaries_dir / "meeting_record.md").write_text(
            "## 議題\n- 該当なし\n", encoding="utf-8"
        )
        captured: dict[str, list[str]] = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            from otoweave_app.llm_chat import _SummaryProcessResult

            return _SummaryProcessResult(0, "", "")

        with patch(
            "otoweave_app.llm_chat._run_summary_process",
            side_effect=fake_run,
        ):
            run_template_summarize_subprocess(
                SimpleNamespace(segments=[]),
                folder,
                Path("C:/fake/project"),
                Path("Qwen3.5-4B-Q4_K_M.gguf"),
                template,
            )
        return captured["command"]

    def test_legacy_pipeline_omits_flag_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with patch("otoweave_app.platform_support.IS_MACOS", False):
                command = self._run_legacy(folder)
        self.assertNotIn("--n_gpu_layers", command)

    def test_legacy_pipeline_adds_flag_on_macos(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with patch("otoweave_app.platform_support.IS_MACOS", True):
                command = self._run_legacy(folder)
        self.assertIn("--n_gpu_layers", command)
        self.assertEqual(command[command.index("--n_gpu_layers") + 1], "-1")

    def test_template_pipeline_omits_flag_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with patch("otoweave_app.platform_support.IS_MACOS", False):
                command = self._run_template(folder)
        self.assertNotIn("--n_gpu_layers", command)

    def test_template_pipeline_adds_flag_on_macos(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with patch("otoweave_app.platform_support.IS_MACOS", True):
                command = self._run_template(folder)
        self.assertIn("--n_gpu_layers", command)
        self.assertEqual(command[command.index("--n_gpu_layers") + 1], "-1")


# ---------------------------------------------------------------------
# Metal (n_gpu_layers=-1) wiring for the resident chat model in
# llm_session.py's chat_async().
# ---------------------------------------------------------------------


class ChatModelMetalArgTests(unittest.TestCase):
    @staticmethod
    def _session() -> tuple[LlmSession, "queue_module.Queue"]:
        events: queue_module.Queue = queue_module.Queue()
        session = LlmSession(Path(__file__).resolve().parent.parent, events)
        return session, events

    def _run_one_chat_turn(self, is_macos: bool) -> dict:
        session, events = self._session()
        captured: dict = {}

        class FakeLlm:
            def __init__(self, *args, **kwargs) -> None:
                captured["kwargs"] = kwargs

        # llm_session.py does `from .platform_support import IS_MACOS` at
        # module import time, so the flag must be patched on llm_session
        # itself -- patching platform_support.IS_MACOS would not reach this
        # already-bound local copy.
        with patch("otoweave_app.llm_session.IS_MACOS", is_macos), patch(
            "llama_cpp.Llama", FakeLlm
        ), patch(
            "otoweave_app.llm_chat.load_context", return_value=""
        ), patch(
            "otoweave_app.llm_chat.build_initial_messages", return_value=[]
        ), patch(
            "otoweave_app.llm_chat.build_retrieval_query", return_value=""
        ), patch(
            "otoweave_app.llm_chat.find_relevant_transcript_excerpts",
            return_value="",
        ), patch(
            "otoweave_app.llm_chat.chat_one_turn",
            return_value=("answer", []),
        ):
            session.chat_async("質問", Path("lesson"), Path("model.gguf"))
            deadline = time.time() + 10
            kinds: list[str] = []
            while time.time() < deadline:
                try:
                    kind, _payload = events.get(timeout=0.5)
                except queue_module.Empty:
                    continue
                kinds.append(kind)
                if kind in {"llm_chat_done", "llm_chat_error"}:
                    break
        self.assertIn("llm_chat_done", kinds, f"chat turn did not finish: {kinds}")
        return captured["kwargs"]

    def test_windows_chat_model_has_no_gpu_layers_argument(self) -> None:
        kwargs = self._run_one_chat_turn(is_macos=False)
        self.assertNotIn("n_gpu_layers", kwargs)

    def test_macos_chat_model_offloads_every_layer_to_metal(self) -> None:
        kwargs = self._run_one_chat_turn(is_macos=True)
        self.assertEqual(kwargs.get("n_gpu_layers"), -1)


# ---------------------------------------------------------------------
# child_popen_kwargs() flowing into the two real subprocess launch sites
# (asr.py's llama-server, llm_chat.py's summarize scripts).
# ---------------------------------------------------------------------


class ChildPopenKwargsWiringTests(unittest.TestCase):
    class FakeServerProcess:
        def __init__(self) -> None:
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    def test_asr_server_popen_gets_new_session_kwarg_on_posix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                "engines/llama-b9763-cpu/llama-server.exe",
                "models/qwen3-asr-gguf/Qwen3-ASR-1.7B-Q8_0.gguf",
                "models/qwen3-asr-gguf/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"stub")

            fake_process = self.FakeServerProcess()
            with patch(
                "otoweave_app.asr.subprocess.Popen",
                return_value=fake_process,
            ) as popen, patch.object(
                Qwen3AsrRecognizer, "_wait_until_ready", return_value=None
            ), patch(
                "otoweave_app.platform_support.IS_WINDOWS", False
            ):
                recognizer = Qwen3AsrRecognizer(root, root / "logs")
                self.assertEqual(
                    popen.call_args.kwargs.get("start_new_session"), True
                )
                recognizer.close()

    def test_asr_server_popen_has_no_new_session_kwarg_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                "engines/llama-b9763-cpu/llama-server.exe",
                "models/qwen3-asr-gguf/Qwen3-ASR-1.7B-Q8_0.gguf",
                "models/qwen3-asr-gguf/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"stub")

            fake_process = self.FakeServerProcess()
            with patch(
                "otoweave_app.asr.subprocess.Popen",
                return_value=fake_process,
            ) as popen, patch.object(
                Qwen3AsrRecognizer, "_wait_until_ready", return_value=None
            ):
                recognizer = Qwen3AsrRecognizer(root, root / "logs")
                self.assertNotIn("start_new_session", popen.call_args.kwargs)
                recognizer.close()

    def test_summary_subprocess_popen_gets_new_session_kwarg_on_posix(self) -> None:
        from otoweave_app.llm_chat import _run_summary_process

        with patch(
            "otoweave_app.llm_chat.subprocess.Popen"
        ) as popen, patch("otoweave_app.platform_support.IS_WINDOWS", False):
            fake_process = MagicMock()
            fake_process.stdout = None
            fake_process.stderr = None
            fake_process.wait.return_value = 0
            fake_process.returncode = 0
            popen.return_value = fake_process
            _run_summary_process(["python", "-c", "pass"], cwd=Path("."))
        self.assertEqual(popen.call_args.kwargs.get("start_new_session"), True)


# ---------------------------------------------------------------------
# POSIX process-group kill (posix_process.py), the counterpart to
# windows_job.py's kill-on-close Job Object.
# ---------------------------------------------------------------------


class PosixProcessKillTests(unittest.TestCase):
    def test_new_session_popen_kwargs(self) -> None:
        self.assertEqual(
            posix_process.new_session_popen_kwargs(), {"start_new_session": True}
        )

    def test_kill_process_group_sends_sigterm_to_the_whole_group(self) -> None:
        process = MagicMock()
        process.poll.return_value = None
        process.pid = 4242
        with patch(
            "os.getpgid", return_value=777, create=True
        ) as getpgid, patch("os.killpg", create=True) as killpg:
            posix_process.kill_process_group(process)
        getpgid.assert_called_once_with(4242)
        killpg.assert_called_once()
        self.assertEqual(killpg.call_args.args[0], 777)

    def test_already_exited_process_is_left_alone(self) -> None:
        process = MagicMock()
        process.poll.return_value = 0
        with patch("os.getpgid", create=True) as getpgid, patch(
            "os.killpg", create=True
        ) as killpg:
            posix_process.kill_process_group(process)
        getpgid.assert_not_called()
        killpg.assert_not_called()

    def test_falls_back_to_plain_kill_when_group_lookup_fails(self) -> None:
        process = MagicMock()
        process.poll.side_effect = [None, None]
        with patch("os.getpgid", side_effect=ProcessLookupError("gone"), create=True):
            posix_process.kill_process_group(process)
        process.kill.assert_called_once()

    def test_llm_session_terminate_uses_process_group_on_posix(self) -> None:
        process = MagicMock()
        with patch("otoweave_app.platform_support.IS_WINDOWS", False), patch(
            "otoweave_app.posix_process.kill_process_group"
        ) as kill_group:
            LlmSession._terminate_process(process)
        kill_group.assert_called_once_with(process)

    def test_llm_session_cancel_summary_kills_registered_process_on_posix(self) -> None:
        events: queue_module.Queue = queue_module.Queue()
        session = LlmSession(Path(__file__).resolve().parent.parent, events)
        process = MagicMock()
        with session._lock:
            session._busy = True
        with patch("otoweave_app.platform_support.IS_WINDOWS", False), patch(
            "otoweave_app.posix_process.kill_process_group"
        ) as kill_group:
            session._register_summary_process(process)
            self.assertTrue(session.cancel_summary())
        kill_group.assert_called_once_with(process)


# ---------------------------------------------------------------------
# --n_gpu_layers CLI argument on the two summarize scripts: default 0
# (CPU only, unchanged Windows behaviour) with a working override.
# ---------------------------------------------------------------------


class SummarizeScriptGpuLayersArgTests(unittest.TestCase):
    def test_template_summarize_defaults_to_cpu_only(self) -> None:
        from scripts.production.template_summarize import parse_args

        with patch(
            "sys.argv",
            [
                "template_summarize.py",
                "--output_dir", "out",
                "--model", "model.gguf",
                "--log", "log.txt",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.n_gpu_layers, 0)

    def test_template_summarize_accepts_metal_override(self) -> None:
        from scripts.production.template_summarize import parse_args

        with patch(
            "sys.argv",
            [
                "template_summarize.py",
                "--output_dir", "out",
                "--model", "model.gguf",
                "--log", "log.txt",
                "--n_gpu_layers", "-1",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.n_gpu_layers, -1)

    def test_school_hybrid_postprocess_defaults_to_cpu_only(self) -> None:
        from scripts.production.school_hybrid_postprocess import parse_args

        with patch(
            "sys.argv",
            [
                "school_hybrid_postprocess.py",
                "--input", "raw.txt",
                "--output_dir", "out",
                "--model", "model.gguf",
                "--log", "log.txt",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.n_gpu_layers, 0)

    def test_school_hybrid_postprocess_accepts_metal_override(self) -> None:
        from scripts.production.school_hybrid_postprocess import parse_args

        with patch(
            "sys.argv",
            [
                "school_hybrid_postprocess.py",
                "--input", "raw.txt",
                "--output_dir", "out",
                "--model", "model.gguf",
                "--log", "log.txt",
                "--n_gpu_layers", "-1",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.n_gpu_layers, -1)


if __name__ == "__main__":
    unittest.main()
