"""Lifecycle of the local LLM (summaries and chat), separated from the
controller.

All busy/release state is read and written under one lock so that a
release requested during inference can never be lost — on 4-8 GB machines
a model that silently stays resident starves the next ASR run.
"""
from __future__ import annotations

import gc
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any

from .windows_job import assign_process_to_job, create_kill_on_close_job


class LlmSession:
    def __init__(
        self,
        project_root: Path,
        events: "queue.Queue[tuple[str, Any]]",
    ) -> None:
        self.project_root = project_root
        self.events = events
        self._llm: Any = None
        self._model_path: Path | None = None
        self._lock = threading.RLock()
        self._busy = False
        self._release_pending = False
        self._chat_messages: list[dict] = []
        self._chat_active_folder: Path | None = None
        # Cancellation state for the summary subprocess.
        self._active_process: subprocess.Popen | None = None
        self._cancel_requested = False
        self._job: int | None = None
        self._job_created = False

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    # ------------------------------------------------------------------
    # Summaries (subprocess-based)
    # ------------------------------------------------------------------

    def _register_summary_process(self, process: subprocess.Popen) -> None:
        """Track the running summary subprocess for cancellation, and put
        it in a kill-on-close job so it dies with the app."""
        with self._lock:
            self._active_process = process
            cancel_now = self._cancel_requested
            if not self._job_created:
                self._job_created = True
                self._job = create_kill_on_close_job()
            job = self._job
        assign_process_to_job(job, process)
        if cancel_now:
            self._terminate_process(process)

    @staticmethod
    def _terminate_process(process: subprocess.Popen) -> None:
        try:
            if process.poll() is None:
                process.kill()
        except OSError:
            pass

    def cancel_summary(self) -> bool:
        """Request cancellation of the running summary. Safe to call when
        nothing is running; the flag also covers the start-up window
        before the subprocess handle is registered."""
        with self._lock:
            if not self._busy:
                return False
            self._cancel_requested = True
            process = self._active_process
        if process is not None:
            self._terminate_process(process)
        return True

    def summarize_async(
        self,
        lesson,
        folder: Path,
        model_path: Path,
        template: dict[str, Any] | None = None,
    ) -> None:
        """Export the transcript and generate a local summary."""
        with self._lock:
            self._busy = True
            self._cancel_requested = False
        self.events.put(("llm_started", "要約を生成中です（数分かかる場合があります）…"))

        project_root = self.project_root

        def _on_progress(progress: dict) -> None:
            # Worker thread -> UI: always go through the events queue.
            self.events.put(("summary_progress", (folder, progress)))

        def _worker() -> None:
            try:
                from . import llm_chat
                if template is None:
                    llm_chat.run_summarize_subprocess(
                        lesson,
                        folder,
                        project_root,
                        model_path,
                        on_process=self._register_summary_process,
                        on_progress=_on_progress,
                    )
                else:
                    llm_chat.run_template_summarize_subprocess(
                        lesson,
                        folder,
                        project_root,
                        model_path,
                        template,
                        on_process=self._register_summary_process,
                        on_progress=_on_progress,
                    )
                self.events.put(("llm_summary_done", folder))
            except Exception as exc:
                with self._lock:
                    cancelled = self._cancel_requested
                if cancelled:
                    self.events.put(("llm_cancelled", folder))
                else:
                    self.events.put(("llm_error", str(exc)))
            finally:
                with self._lock:
                    self._active_process = None
                self._finish_task()

        threading.Thread(target=_worker, daemon=True, name="llm-summarize").start()

    # ------------------------------------------------------------------
    # Chat Q&A (resident model)
    # ------------------------------------------------------------------

    def chat_async(
        self,
        question: str,
        lesson_folder: Path,
        model_path: Path,
    ) -> None:
        """Answer a question about the current session in a background thread."""
        with self._lock:
            self._busy = True
            self._chat_active_folder = lesson_folder
        self.events.put(("llm_chat_thinking", None))

        def _worker() -> None:
            try:
                from . import llm_chat
                from .asr import select_asr_threads
                with self._lock:
                    if self._llm is None or self._model_path != model_path:
                        self._release_locked()
                        from llama_cpp import Llama  # type: ignore[import-not-found]
                        thread_config = select_asr_threads("file")
                        self._llm = Llama(
                            model_path=str(model_path),
                            n_ctx=4096,
                            n_threads=thread_config.num_threads,
                            n_batch=256,
                            verbose=False,
                        )
                        self._model_path = model_path
                if not self._chat_messages:
                    context = llm_chat.load_context(lesson_folder)
                    self._chat_messages = llm_chat.build_initial_messages(context)
                retrieval_query = llm_chat.build_retrieval_query(
                    self._chat_messages,
                    question,
                )
                relevant_excerpts = llm_chat.find_relevant_transcript_excerpts(
                    lesson_folder,
                    retrieval_query,
                )
                answer, updated_messages = llm_chat.chat_one_turn(
                    self._llm,
                    self._chat_messages,
                    question,
                    relevant_excerpts=relevant_excerpts,
                    # どのノートへの回答かを含めて送り、受信側で現在の
                    # ノートと一致するときだけ画面へ追記できるようにする。
                    on_chunk=lambda text: self.events.put(
                        ("llm_chat_chunk", (text, lesson_folder))
                    ),
                )
                with self._lock:
                    if self._chat_active_folder == lesson_folder:
                        self._chat_messages = updated_messages
                self.events.put(("llm_chat_done", (answer, lesson_folder)))
            except Exception as exc:
                self.events.put(("llm_chat_error", (str(exc), lesson_folder)))
            finally:
                self._finish_task()

        threading.Thread(target=_worker, daemon=True, name="llm-chat").start()

    def reset_chat(self) -> None:
        """Clear conversation history so the next question starts fresh."""
        with self._lock:
            self._chat_messages = []
            self._chat_active_folder = None

    def release_model(self) -> bool:
        """Release the resident chat model, or defer release until inference ends."""
        with self._lock:
            if self._busy:
                self._release_pending = True
                return False
            self._release_locked()
            return True

    def _finish_task(self) -> None:
        with self._lock:
            self._busy = False
            if self._release_pending:
                self._release_locked()

    def _release_locked(self) -> None:
        llm = self._llm
        self._llm = None
        self._model_path = None
        self._chat_messages = []
        self._release_pending = False
        if llm is None:
            return
        close = getattr(llm, "close", None)
        if callable(close):
            close()
        del llm
        gc.collect()
