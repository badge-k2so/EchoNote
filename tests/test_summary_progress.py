"""Tests for live summary progress streaming and RAM-based profiles."""
import contextlib
import io
import json
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from otoweave_app.llm_chat import (
    SUMMARIZE_MODEL_9B_FILENAME,
    _parse_progress_line,
    _run_summary_process,
    find_summarize_model,
    run_summarize_subprocess,
    run_template_summarize_subprocess,
    summarize_llm_profile,
)
from scripts.production.template_summarize import print_progress


class ParseProgressLineTests(unittest.TestCase):
    def test_parses_part_progress(self) -> None:
        line = '{"progress": {"stage": "part", "current": 2, "total": 5}}\n'
        self.assertEqual(
            _parse_progress_line(line),
            {"stage": "part", "current": 2, "total": 5},
        )

    def test_parses_merge_progress(self) -> None:
        self.assertEqual(
            _parse_progress_line('{"progress": {"stage": "merge"}}'),
            {"stage": "merge"},
        )

    def test_plain_log_line_is_not_progress(self) -> None:
        self.assertIsNone(_parse_progress_line("Summarizing template part 1/3"))

    def test_json_without_progress_key_is_not_progress(self) -> None:
        self.assertIsNone(_parse_progress_line('{"template": "lesson_record"}'))

    def test_invalid_json_is_not_progress(self) -> None:
        self.assertIsNone(_parse_progress_line('{"progress": {'))

    def test_non_dict_progress_is_not_progress(self) -> None:
        self.assertIsNone(_parse_progress_line('{"progress": "part"}'))

    def test_script_helper_output_round_trips(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print_progress("part", current=3, total=7)
            print_progress("merge")
        lines = buffer.getvalue().splitlines()
        self.assertEqual(
            _parse_progress_line(lines[0]),
            {"stage": "part", "current": 3, "total": 7},
        )
        self.assertEqual(_parse_progress_line(lines[1]), {"stage": "merge"})


class RunSummaryProcessTests(unittest.TestCase):
    """Exercise the streaming runner with small real subprocesses."""

    @staticmethod
    def _python_command(code: str) -> list[str]:
        return [sys.executable, "-X", "utf8", "-c", code]

    def test_progress_lines_reach_callback_and_stay_out_of_stdout(self) -> None:
        code = (
            "import json, sys\n"
            "print('Loading model: fake', flush=True)\n"
            "print(json.dumps({'progress': {'stage': 'part', 'current': 1, 'total': 2}}), flush=True)\n"
            "print(json.dumps({'progress': {'stage': 'part', 'current': 2, 'total': 2}}), flush=True)\n"
            "print(json.dumps({'progress': {'stage': 'merge'}}), flush=True)\n"
            "print('done', flush=True)\n"
            "print('warning line', file=sys.stderr, flush=True)\n"
        )
        progress_events: list[dict] = []
        result = _run_summary_process(
            self._python_command(code),
            cwd=Path.cwd(),
            on_progress=progress_events.append,
            idle_timeout=30.0,
            total_timeout=60.0,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            progress_events,
            [
                {"stage": "part", "current": 1, "total": 2},
                {"stage": "part", "current": 2, "total": 2},
                {"stage": "merge"},
            ],
        )
        self.assertIn("Loading model: fake", result.stdout)
        self.assertIn("done", result.stdout)
        self.assertNotIn("progress", result.stdout)
        self.assertIn("warning line", result.stderr)

    def test_works_without_progress_callback(self) -> None:
        code = (
            "import json\n"
            "print(json.dumps({'progress': {'stage': 'merge'}}), flush=True)\n"
            "print('finished', flush=True)\n"
        )
        result = _run_summary_process(
            self._python_command(code),
            cwd=Path.cwd(),
            idle_timeout=30.0,
            total_timeout=60.0,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("finished", result.stdout)

    def test_progress_callback_error_does_not_break_run(self) -> None:
        code = (
            "import json\n"
            "print(json.dumps({'progress': {'stage': 'merge'}}), flush=True)\n"
            "print('finished', flush=True)\n"
        )

        def broken_callback(_: dict) -> None:
            raise ValueError("callback error")

        result = _run_summary_process(
            self._python_command(code),
            cwd=Path.cwd(),
            on_progress=broken_callback,
            idle_timeout=30.0,
            total_timeout=60.0,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("finished", result.stdout)

    def test_idle_timeout_kills_silent_process(self) -> None:
        code = "import time\ntime.sleep(60)\n"
        with self.assertRaises(RuntimeError) as raised:
            _run_summary_process(
                self._python_command(code),
                cwd=Path.cwd(),
                idle_timeout=0.5,
                total_timeout=60.0,
            )
        self.assertIn("タイムアウト", str(raised.exception))

    def test_output_resets_idle_timeout(self) -> None:
        # Prints every 0.2s for ~1.2s: far longer than the 0.6s idle limit,
        # but each line resets the idle clock so the run must succeed.
        code = (
            "import time\n"
            "for i in range(6):\n"
            "    print(f'line {i}', flush=True)\n"
            "    time.sleep(0.2)\n"
        )
        result = _run_summary_process(
            self._python_command(code),
            cwd=Path.cwd(),
            idle_timeout=0.6,
            total_timeout=60.0,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("line 5", result.stdout)

    def test_total_timeout_kills_chatty_process(self) -> None:
        code = (
            "import time\n"
            "for i in range(600):\n"
            "    print(f'line {i}', flush=True)\n"
            "    time.sleep(0.1)\n"
        )
        with self.assertRaises(RuntimeError) as raised:
            _run_summary_process(
                self._python_command(code),
                cwd=Path.cwd(),
                idle_timeout=30.0,
                total_timeout=0.8,
            )
        self.assertIn("タイムアウト", str(raised.exception))

    def test_external_kill_returns_nonzero_without_timeout_error(self) -> None:
        # Cancellation kills the process from outside; the runner must
        # return the non-zero exit code instead of raising a timeout.
        code = "import time\ntime.sleep(60)\n"
        handle: list = []

        def on_process(process) -> None:
            handle.append(process)
            process.kill()

        result = _run_summary_process(
            self._python_command(code),
            cwd=Path.cwd(),
            on_process=on_process,
            idle_timeout=30.0,
            total_timeout=60.0,
        )
        self.assertEqual(len(handle), 1)
        self.assertNotEqual(result.returncode, 0)


class RamProfileTests(unittest.TestCase):
    """Real 8GB machines report ~7.9GB and count as low-memory.

    New policy: summarization is 4B-only. Low-memory machines get no
    summarize model at all (None) instead of a 2B fallback; the low-memory
    context profile is kept only for future lightweight models."""

    @staticmethod
    def _make_models(root: Path) -> None:
        (root / "models").mkdir()
        (root / "models" / "Qwen3.5-4B-Q4_K_M.gguf").write_bytes(b"x")
        (root / "models" / "Qwen3.5-2B-Q4_K_M.gguf").write_bytes(b"x")

    def _profile_with_ram(self, ram_bytes: int) -> dict:
        with patch(
            "otoweave_app.llm_chat._total_physical_ram_bytes",
            return_value=ram_bytes,
        ):
            return summarize_llm_profile()

    def _model_with_ram(self, ram_bytes: int) -> Path | None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_models(root)
            with patch(
                "otoweave_app.llm_chat._total_physical_ram_bytes",
                return_value=ram_bytes,
            ):
                return find_summarize_model(root)

    def test_real_8gb_machine_has_no_summarize_model(self) -> None:
        ram = int(7.9 * 1024**3)
        profile = self._profile_with_ram(ram)
        self.assertEqual(profile["n_ctx"], 4096)
        # 低RAM機では2Bへフォールバックせず、要約自体を非対応にする。
        self.assertIsNone(self._model_with_ram(ram))

    def test_16gb_machine_uses_high_profile_and_4b(self) -> None:
        ram = 16 * 1024**3
        profile = self._profile_with_ram(ram)
        self.assertEqual(profile["n_ctx"], 8192)
        self.assertIn("4B", self._model_with_ram(ram).name)

    def test_ram_query_failure_falls_back_to_low_memory(self) -> None:
        profile = self._profile_with_ram(0)
        self.assertEqual(profile["n_ctx"], 4096)
        # RAM取得に失敗したときも安全側（要約非対応）に倒す。
        self.assertIsNone(self._model_with_ram(0))

    def test_9b_model_gets_wider_token_budgets(self) -> None:
        # A/Bベンチで、9Bはマージ段階の自発的な過剰圧縮により網羅性が
        # 落ちることが判明したため、9B選択時だけmax_tokens_part/finalを
        # 広げる（n_ctx/n_threads/n_batchは4Bと共通のまま）。
        model_path = Path("C:/fake/models") / SUMMARIZE_MODEL_9B_FILENAME
        profile = summarize_llm_profile(model_path)
        self.assertEqual(profile["n_ctx"], 8192)
        self.assertEqual(profile["max_tokens_part"], 1500)
        self.assertEqual(profile["max_tokens_final"], 2400)

    def test_4b_model_keeps_original_budgets(self) -> None:
        model_path = Path("C:/fake/models/Qwen3.5-4B-Q4_K_M.gguf")
        profile = summarize_llm_profile(model_path)
        self.assertNotIn("max_tokens_part", profile)
        self.assertEqual(profile["max_tokens_final"], 1800)

    def test_no_model_path_keeps_original_budgets(self) -> None:
        profile = summarize_llm_profile()
        self.assertNotIn("max_tokens_part", profile)
        self.assertEqual(profile["max_tokens_final"], 1800)


class SummarizeSubprocessCommandTests(unittest.TestCase):
    """The subprocess launchers must pass the 9B token overrides only when
    the 9B model is selected, leaving the 4B command line unchanged."""

    def _run_legacy(self, model_path: Path, folder: Path) -> list[str]:
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
                model_path,
            )
        return captured["command"]

    def _run_template(self, model_path: Path, folder: Path) -> list[str]:
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
                model_path,
                template,
            )
        return captured["command"]

    def test_legacy_pipeline_adds_9b_overrides_only_for_9b(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder_4b = Path(temporary) / "lesson_4b"
            folder_4b.mkdir()
            command_4b = self._run_legacy(
                Path("C:/fake/models/Qwen3.5-4B-Q4_K_M.gguf"), folder_4b
            )
            folder_9b = Path(temporary) / "lesson_9b"
            folder_9b.mkdir()
            command_9b = self._run_legacy(
                Path("C:/fake/models") / SUMMARIZE_MODEL_9B_FILENAME, folder_9b
            )
        self.assertNotIn("--max_tokens_part", command_4b)
        self.assertNotIn("--max_tokens_final", command_4b)
        self.assertIn("--max_tokens_part", command_9b)
        part_index = command_9b.index("--max_tokens_part")
        self.assertEqual(command_9b[part_index + 1], "1500")
        final_index = command_9b.index("--max_tokens_final")
        self.assertEqual(command_9b[final_index + 1], "2400")

    def test_template_pipeline_adds_max_tokens_part_only_for_9b(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder_4b = Path(temporary) / "lesson_4b"
            folder_4b.mkdir()
            command_4b = self._run_template(
                Path("C:/fake/models/Qwen3.5-4B-Q4_K_M.gguf"), folder_4b
            )
            folder_9b = Path(temporary) / "lesson_9b"
            folder_9b.mkdir()
            command_9b = self._run_template(
                Path("C:/fake/models") / SUMMARIZE_MODEL_9B_FILENAME, folder_9b
            )
        self.assertNotIn("--max_tokens_part", command_4b)
        final_index_4b = command_4b.index("--max_tokens_final")
        self.assertEqual(command_4b[final_index_4b + 1], "1800")

        self.assertIn("--max_tokens_part", command_9b)
        part_index = command_9b.index("--max_tokens_part")
        self.assertEqual(command_9b[part_index + 1], "1500")
        final_index_9b = command_9b.index("--max_tokens_final")
        self.assertEqual(command_9b[final_index_9b + 1], "2400")


class SessionProgressEventTests(unittest.TestCase):
    def test_summary_progress_events_reach_queue(self) -> None:
        from otoweave_app.llm_session import LlmSession

        events: "queue.Queue" = queue.Queue()
        folder = Path("C:/fake/lesson")
        session = LlmSession(Path("C:/fake/project"), events)

        def fake_run(lesson, lesson_folder, project_root, model_path,
                     on_process=None, on_progress=None) -> None:
            if on_progress is not None:
                on_progress({"stage": "part", "current": 1, "total": 3})
                on_progress({"stage": "merge"})

        with patch(
            "otoweave_app.llm_chat.run_summarize_subprocess",
            side_effect=fake_run,
        ):
            session.summarize_async(
                SimpleNamespace(segments=[]),
                folder,
                Path("C:/fake/model.gguf"),
            )
            received: list[tuple] = []
            while True:
                name, payload = events.get(timeout=5)
                received.append((name, payload))
                if name in ("llm_summary_done", "llm_error"):
                    break

        names = [name for name, _ in received]
        self.assertIn("summary_progress", names)
        self.assertEqual(names[-1], "llm_summary_done")
        progress_payloads = [
            payload for name, payload in received if name == "summary_progress"
        ]
        self.assertEqual(
            progress_payloads,
            [
                (folder, {"stage": "part", "current": 1, "total": 3}),
                (folder, {"stage": "merge"}),
            ],
        )


if __name__ == "__main__":
    unittest.main()
