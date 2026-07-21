from __future__ import annotations

import base64
import io
import json
import os
import queue
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .app_logging import log_exception
from .audio import SAMPLE_RATE, SpeechChunk
from .windows_job import assign_process_to_job, create_kill_on_close_job


QWEN3_ASR_17_MODEL = Path("models/qwen3-asr-gguf/Qwen3-ASR-1.7B-Q8_0.gguf")
QWEN3_ASR_17_MMPROJ = Path("models/qwen3-asr-gguf/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf")
QWEN3_ASR_SERVER = Path("engines/llama-b9763-cpu/llama-server.exe")
SPEECHBRAIN_LANGUAGE_ID_DIR = Path("models/speechbrain-lang-id-voxlingua107-ecapa-onnx")


@dataclass(frozen=True)
class AsrThreadConfig:
    processing_mode: str
    detected_logical_cpus: int
    num_threads: int


def select_asr_threads(
    processing_mode: str,
    logical_cpus: int | None = None,
) -> AsrThreadConfig:
    if processing_mode not in {"live", "file"}:
        raise ValueError(f"Unknown ASR processing mode: {processing_mode}")
    detected = max(1, logical_cpus if logical_cpus is not None else (os.cpu_count() or 1))
    limit = 2 if processing_mode == "live" else 4
    return AsrThreadConfig(processing_mode, detected, min(limit, detected))


@dataclass(frozen=True)
class RecognizedSentence:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class LanguageDecision:
    language: str
    confidence: float
    margin: float
    japanese_score: float
    english_score: float
    uncertain: bool
    reason: str


class SpeechBrainLanguageIdentifier:
    MIN_SAMPLES = SAMPLE_RATE
    UNCERTAIN_CONFIDENCE = 0.15
    UNCERTAIN_MARGIN = 0.10

    def __init__(self, model_dir: Path, num_threads: int = 2) -> None:
        import onnxruntime as ort

        model_dir = Path(model_dir)
        model_path = model_dir / "lang-id-ecapa.onnx"
        labels_path = model_dir / "labels.json"
        missing = [str(path) for path in (model_path, labels_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("SpeechBrain language ID files are missing: " + ", ".join(missing))

        self.labels = json.loads(labels_path.read_text(encoding="utf-8"))
        self.japanese_index = self.labels.index("ja")
        self.english_index = self.labels.index("en")
        self.num_threads = max(1, int(num_threads))
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.num_threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def detect(self, samples: np.ndarray) -> LanguageDecision:
        waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
        if waveform.size and float(np.max(np.abs(waveform))) > 1.5:
            waveform = waveform / 32768.0
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > np.finfo(np.float32).eps:
            waveform = waveform * (0.95 / peak)

        probabilities = np.asarray(
            self.session.run(None, {"waveform": waveform.reshape(1, -1)})[0],
            dtype=np.float32,
        ).reshape(-1)
        japanese_score = float(probabilities[self.japanese_index])
        english_score = float(probabilities[self.english_index])
        raw_language = "english" if english_score > japanese_score else "japanese"
        confidence = max(japanese_score, english_score)
        margin = abs(english_score - japanese_score)
        # Short speech is less reliable, but a strong English result should not
        # be discarded solely because the utterance is under one second.
        prefer_japanese = margin < self.UNCERTAIN_MARGIN
        language = "japanese" if prefer_japanese else raw_language
        if language == "english":
            language = "english"
            confidence = english_score
        else:
            confidence = japanese_score

        reasons: list[str] = []
        if waveform.size < self.MIN_SAMPLES:
            reasons.append("short_audio")
        if confidence < self.UNCERTAIN_CONFIDENCE:
            reasons.append("low_confidence")
        if margin < self.UNCERTAIN_MARGIN:
            reasons.append("close_ja_en_scores")
        if prefer_japanese and raw_language == "english":
            reasons.append("japanese_default_route")
        return LanguageDecision(
            language=language,
            confidence=confidence,
            margin=margin,
            japanese_score=japanese_score,
            english_score=english_score,
            uncertain=bool(reasons),
            reason=",".join(reasons),
        )


def split_text_with_times(text: str, start: float, end: float, max_chars: int = 72) -> list[RecognizedSentence]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = [part.strip() for part in re.findall(r".*?(?:[。！？!?]+|\.(?=\s|$)|$)", normalized) if part.strip()]
    pieces: list[str] = []
    for sentence in sentences or [normalized]:
        remaining = sentence
        while len(remaining) > max_chars:
            boundary = max(
                remaining.rfind(" ", 0, max_chars + 1),
                remaining.rfind("、", 0, max_chars + 1),
                remaining.rfind(",", 0, max_chars + 1),
            )
            if boundary < max_chars // 2:
                boundary = max_chars
            pieces.append(remaining[:boundary].strip(" 、,"))
            remaining = remaining[boundary:].strip(" 、,")
        if remaining:
            pieces.append(remaining)

    total_weight = sum(max(1, len(piece)) for piece in pieces)
    duration = max(0.1, end - start)
    cursor = start
    result: list[RecognizedSentence] = []
    for index, piece in enumerate(pieces):
        piece_duration = duration * max(1, len(piece)) / total_weight
        piece_end = end if index == len(pieces) - 1 else min(end, cursor + piece_duration)
        result.append(RecognizedSentence(cursor, piece_end, piece))
        cursor = piece_end
    return result


class JapaneseRecognizer:
    def __init__(self, language: str = "ja", num_threads: int = 2) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        if language != "ja":
            raise ValueError(f"Unsupported Japanese ASR language: {language}")

        import sherpa_onnx
        from huggingface_hub import snapshot_download

        model_dir = Path(
            snapshot_download(
                "reazon-research/reazonspeech-k2-v2",
                local_files_only=True,
            )
        )
        paths = {
            "encoder": model_dir / "encoder-epoch-99-avg-1.int8.onnx",
            "decoder": model_dir / "decoder-epoch-99-avg-1.int8.onnx",
            "joiner": model_dir / "joiner-epoch-99-avg-1.int8.onnx",
            "tokens": model_dir / "tokens.txt",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("ReazonSpeech model files are missing: " + ", ".join(missing))

        self.num_threads = max(1, int(num_threads))
        self.model = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(paths["encoder"]),
            decoder=str(paths["decoder"]),
            joiner=str(paths["joiner"]),
            tokens=str(paths["tokens"]),
            num_threads=self.num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu",
        )

    def transcribe(self, samples: np.ndarray) -> str:
        from reazonspeech.k2.asr import audio_from_numpy, transcribe

        audio = audio_from_numpy(samples.astype(np.float32) / 32768.0, SAMPLE_RATE)
        return str(getattr(transcribe(self.model, audio), "text", "")).strip()


class EnglishRecognizer:
    def __init__(self, model_dir: Path, num_threads: int = 2) -> None:
        import sherpa_onnx

        model_dir = Path(model_dir)
        paths = {
            "encoder": model_dir / "encoder.int8.onnx",
            "decoder": model_dir / "decoder.int8.onnx",
            "joiner": model_dir / "joiner.int8.onnx",
            "tokens": model_dir / "tokens.txt",
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Parakeet model files are missing: " + ", ".join(missing))
        self.num_threads = max(1, int(num_threads))
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(paths["encoder"]),
            decoder=str(paths["decoder"]),
            joiner=str(paths["joiner"]),
            tokens=str(paths["tokens"]),
            num_threads=self.num_threads,
            decoding_method="greedy_search",
            provider="cpu",
            model_type="nemo_transducer",
        )

    def transcribe(self, samples: np.ndarray) -> str:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples.astype(np.float32) / 32768.0)
        self.recognizer.decode_stream(stream)
        return str(getattr(stream.result, "text", "")).strip()


_asr_job_lock = threading.Lock()
_asr_job: int | None = None
_asr_job_created = False


def _asr_kill_on_close_job() -> int | None:
    """One kill-on-close Job Object shared by all ASR server processes.

    The handle is intentionally never closed: it must live as long as this
    process so a force-closed app takes the llama-server down with it."""
    global _asr_job, _asr_job_created
    with _asr_job_lock:
        if not _asr_job_created:
            _asr_job_created = True
            _asr_job = create_kill_on_close_job()
        return _asr_job


def qwen3_asr_17_available(project_root: Path) -> bool:
    root = Path(project_root)
    server = root / QWEN3_ASR_SERVER
    model = root / QWEN3_ASR_17_MODEL
    mmproj = root / QWEN3_ASR_17_MMPROJ
    return (
        server.is_file()
        and model.is_file()
        and model.stat().st_size > 1_000_000_000
        and mmproj.is_file()
        and mmproj.stat().st_size > 100_000_000
    )


class Qwen3AsrRecognizer:
    """Optional multilingual ASR served by the local llama.cpp audio endpoint."""

    def __init__(self, project_root: Path, log_dir: Path, num_threads: int | None = None) -> None:
        self.project_root = Path(project_root)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.server = self.project_root / QWEN3_ASR_SERVER
        self.model = self.project_root / QWEN3_ASR_17_MODEL
        self.mmproj = self.project_root / QWEN3_ASR_17_MMPROJ
        missing = [str(path) for path in (self.server, self.model, self.mmproj) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Qwen3-ASR-1.7B files are missing: " + ", ".join(missing))

        self.port = self._free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._stdout = (self.log_dir / "qwen3_asr_server_stdout.log").open("w", encoding="utf-8")
        self._stderr = (self.log_dir / "qwen3_asr_server_stderr.log").open("w", encoding="utf-8")
        command = [
            str(self.server),
            "-m", str(self.model),
            "-mm", str(self.mmproj),
            "-c", "8192",
            "-t", str(num_threads or select_asr_threads("file").num_threads),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--no-webui",
            "--no-warmup",
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        self.process: subprocess.Popen | None = None
        try:
            self.process = subprocess.Popen(
                command,
                stdout=self._stdout,
                stderr=self._stderr,
                creationflags=creationflags,
            )
        except Exception:
            self.close()
            raise
        # Register with the kill-on-close job right after start: if the
        # app is force-closed the OS terminates the server, so it can no
        # longer survive as an orphan occupying CPU and RAM.
        assign_process_to_job(_asr_kill_on_close_job(), self.process)
        try:
            self._wait_until_ready()
        except Exception:
            self.close()
            raise

    def transcribe(self, samples: np.ndarray) -> str:
        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(audio.getvalue()).decode("ascii"),
                                "format": "wav",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Transcribe the audio faithfully. Output only the transcript.",
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 512,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Qwen3-ASR request failed: {details}") from exc
        content = str(result["choices"][0]["message"]["content"])
        match = re.search(
            r"<asr_text>(.*?)(?:</asr_text>|$)",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return (match.group(1) if match else content).strip()

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self.process = None
        for stream_name in ("_stdout", "_stderr"):
            stream = getattr(self, stream_name, None)
            if stream is not None and not stream.closed:
                stream.close()

    def _wait_until_ready(self, timeout: float = 180.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Qwen3-ASR server exited during startup: {self.process.returncode}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2):
                    return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.25)
        raise TimeoutError("Qwen3-ASR server startup timed out")

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


class LiveAsrWorker:
    # Up to ~7 minutes of speech backlog (7 s chunks); on a slow CPU the
    # recognition can be slower than real time and an unbounded queue would
    # grow for the whole lesson.
    MAX_PENDING_CHUNKS = 60

    def __init__(
        self,
        mode: str,
        project_root: Path,
        on_sentences: Callable[[list[RecognizedSentence]], None],
        on_status: Callable[[str], None],
        on_error: Callable[[str], None],
        num_threads: int | None = None,
    ) -> None:
        self.mode = mode
        self.project_root = Path(project_root)
        self.on_sentences = on_sentences
        self.on_status = on_status
        self.on_error = on_error
        self.num_threads = num_threads or select_asr_threads("live").num_threads
        self._queue: queue.Queue[SpeechChunk | None] = queue.Queue(
            maxsize=self.MAX_PENDING_CHUNKS
        )
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        # Once closed, submit() is a no-op and results are no longer
        # delivered — a worker that outlives its stop timeout must not
        # inject text into the next lesson.
        self._closed = threading.Event()
        self._last_drop_notice = 0.0
        self._last_backlog_notice = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"asr-{self.mode}", daemon=True)
        self._thread.start()

    def submit(self, chunk: SpeechChunk) -> None:
        if self._closed.is_set():
            return
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            now = time.monotonic()
            if now - self._last_drop_notice >= 10.0:
                self._last_drop_notice = now
                self.on_error(
                    "文字起こしが追いつかないため、一部の区間をリアルタイム表示から省略しました。"
                    "音声は保存されているので、あとから文字起こしで補完できます。"
                )

    def stop(self, timeout: float = 120.0) -> None:
        self._signal_stop()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                self._closed.set()
                self.on_error(
                    "残りの文字起こしが時間内に終わらなかったため打ち切りました。"
                    "音声は保存されているので、あとから文字起こしで補完できます。"
                )
        self._closed.set()

    def _signal_stop(self) -> None:
        """Append the stop sentinel without ever deadlocking.

        The queue is bounded: give a live worker a moment to make room so
        the backlog is still transcribed, but if it stays full (worker dead
        or stuck) discard one pending chunk instead of blocking forever."""
        while True:
            try:
                self._queue.put(None, timeout=5.0)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

    def _report_backlog(self) -> None:
        pending = self._queue.qsize()
        if pending < 5:
            return
        now = time.monotonic()
        if now - self._last_backlog_notice >= 10.0:
            self._last_backlog_notice = now
            self.on_status(f"文字起こし中（未処理 {pending} 区間・遅れあり）")

    def _run(self) -> None:
        try:
            self.on_status("音声認識モデルを準備中")
            if self.mode == "japanese":
                recognizer = JapaneseRecognizer(num_threads=self.num_threads)
            elif self.mode == "english":
                recognizer = EnglishRecognizer(
                    self.project_root / "models" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
                    num_threads=self.num_threads,
                )
            else:
                return
            self._ready.set()
            self.on_status("文字起こし中")
            while not self._closed.is_set():
                chunk = self._queue.get()
                if chunk is None:
                    break
                try:
                    text = recognizer.transcribe(chunk.samples)
                    sentences = split_text_with_times(text, chunk.start, chunk.end)
                    if sentences and not self._closed.is_set():
                        self.on_sentences(sentences)
                except Exception as exc:
                    log_exception("ライブ文字起こしの一部区間に失敗", exc)
                    self.on_error("一部の音声を文字にできませんでした。録音はそのまま続いています。")
                self._report_backlog()
            if not self._closed.is_set():
                self.on_status("文字起こし完了")
        except Exception as exc:
            self._ready.set()
            # Stop accepting chunks: without a consumer the bounded queue
            # would only fill up and spam drop warnings.
            self._closed.set()
            log_exception("ライブ文字起こしモデルの読み込みに失敗", exc)
            self.on_error(
                "文字起こしの準備ができませんでした。録音はそのまま続いています。"
                "録音を保存したあと「あとから文字起こし」をお試しください。"
            )
