from __future__ import annotations

import math
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .app_logging import log_exception


SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AudioSource:
    id: str
    label: str
    device_index: int
    sample_rate: int
    channels: int
    kind: str


@dataclass(frozen=True)
class AudioProcessingOptions:
    noise_reduction: bool = False
    sensitivity: float = 1.0
    automatic_gain_control: bool = False


@dataclass(frozen=True)
class SpeechChunk:
    start: float
    end: float
    samples: np.ndarray


def measure_audio_input(source: AudioSource, duration_seconds: float = 1.5) -> dict[str, float | str]:
    """Measure a short input sample without retaining audio."""
    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    stream = None
    try:
        frames = max(320, source.sample_rate // 10)
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=source.channels,
            rate=source.sample_rate,
            input=True,
            input_device_index=source.device_index,
            frames_per_buffer=frames,
        )
        blocks = []
        reads = max(1, int(duration_seconds * source.sample_rate / frames))
        for _ in range(reads):
            blocks.append(stream.read(frames, exception_on_overflow=False))
        samples = np.frombuffer(b"".join(blocks), dtype=np.int16).astype(np.float32)
        if source.channels > 1 and samples.size:
            samples = samples.reshape(-1, source.channels).mean(axis=1)
        normalized = samples / 32768.0
        rms = float(np.sqrt(np.mean(normalized * normalized) + 1e-12))
        peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
        if rms < 0.004 or peak > 0.99:
            state = "Poor"
        elif rms < 0.012 or peak > 0.92:
            state = "Caution"
        else:
            state = "Good"
        return {"state": state, "rms": rms, "peak": peak}
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()


def available_audio_sources() -> list[AudioSource]:
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        return []

    pa = pyaudio.PyAudio()
    sources: list[AudioSource] = []
    seen: set[int] = set()
    try:
        default_input = -1
        default_loopback = -1
        try:
            default_input = int(pa.get_default_input_device_info()["index"])
        except OSError:
            pass
        try:
            default_loopback = int(pa.get_default_wasapi_loopback()["index"])
        except OSError:
            pass

        preferred = [default_input, default_loopback]
        indexes = preferred + [
            index
            for index in range(pa.get_device_count())
            if index not in preferred
        ]
        for index in indexes:
            if index < 0 or index in seen:
                continue
            try:
                info = pa.get_device_info_by_index(index)
            except OSError:
                continue
            is_loopback = bool(info.get("isLoopbackDevice", False))
            max_inputs = int(info.get("maxInputChannels", 0))
            if max_inputs <= 0 and not is_loopback:
                continue
            seen.add(index)
            kind = "loopback" if is_loopback else "microphone"
            prefix = "PC音声" if is_loopback else "マイク"
            default_mark = "（既定）" if index in preferred else ""
            name = str(info.get("name", f"Device {index}"))
            sources.append(
                AudioSource(
                    id=f"{kind}:{index}",
                    label=f"{prefix}{default_mark}: {name}",
                    device_index=index,
                    sample_rate=max(1, int(info.get("defaultSampleRate", SAMPLE_RATE))),
                    channels=max(1, min(2, max_inputs)),
                    kind=kind,
                )
            )
    finally:
        pa.terminate()
    return sources


def process_audio_samples(
    samples: np.ndarray,
    options: AudioProcessingOptions,
) -> np.ndarray:
    """Apply lightweight local input processing to one mono audio block."""
    if samples.size == 0:
        return samples.astype(np.int16, copy=False)
    values = samples.astype(np.float32)
    normalized = values / 32768.0
    rms = float(np.sqrt(np.mean(normalized * normalized) + 1e-12))

    if options.noise_reduction and rms < 0.012:
        values *= max(0.08, min(1.0, rms / 0.012))

    values *= max(0.25, min(2.0, options.sensitivity))
    if options.automatic_gain_control and rms > 0.001:
        agc_gain = max(0.5, min(4.0, 0.08 / rms))
        values *= agc_gain

    return np.clip(np.rint(values), -32768, 32767).astype(np.int16)


class AdaptiveVad:
    """Small energy VAD that retains absolute positions in the source stream."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        pre_roll_seconds: float = 0.4,
        end_silence_seconds: float = 1.0,
        max_chunk_seconds: float = 7.0,
        min_chunk_seconds: float = 0.8,
    ) -> None:
        self.sample_rate = sample_rate
        self.pre_roll_seconds = pre_roll_seconds
        self.end_silence_seconds = end_silence_seconds
        self.max_chunk_seconds = max_chunk_seconds
        self.min_chunk_seconds = min_chunk_seconds
        self.noise_rms = 0.004
        self.pre_roll: deque[tuple[float, np.ndarray]] = deque()
        self.active: list[tuple[float, np.ndarray]] = []
        self.trailing_silence = 0.0

    def reset(self) -> None:
        self.pre_roll.clear()
        self.active.clear()
        self.trailing_silence = 0.0

    def process(self, samples: np.ndarray, start: float) -> list[SpeechChunk]:
        if samples.size == 0:
            return []
        float_samples = samples.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(float_samples * float_samples) + 1e-12))
        duration = samples.size / self.sample_rate
        threshold = max(0.008, self.noise_rms * 2.8)
        voiced = rms >= threshold

        if not self.active:
            if not voiced:
                self.noise_rms = self.noise_rms * 0.97 + rms * 0.03
            self.pre_roll.append((start, samples.copy()))
            self._trim_pre_roll(start + duration)
            if voiced:
                self.active = list(self.pre_roll)
                self.pre_roll.clear()
            return []

        self.active.append((start, samples.copy()))
        self.trailing_silence = 0.0 if voiced else self.trailing_silence + duration
        active_duration = self._active_duration()
        if self.trailing_silence >= self.end_silence_seconds or active_duration >= self.max_chunk_seconds:
            return self.flush()
        return []

    def flush(self, force: bool = False) -> list[SpeechChunk]:
        """Emit the active speech chunk.

        Chunks shorter than min_chunk_seconds are not discarded: they are
        carried over as pre-roll so speech that resumes right after is
        kept. With force=True (pause/stop/end of file) even a very short
        utterance such as 「はい」 is emitted so it reaches the transcript."""
        if not self.active:
            return []
        start = self.active[0][0]
        samples = np.concatenate([block for _, block in self.active])
        end = start + samples.size / self.sample_rate
        if end - start < self.min_chunk_seconds and not force:
            self.pre_roll.extend(self.active)
            self.active = []
            self.trailing_silence = 0.0
            self._trim_pre_roll(end)
            return []
        self.active = []
        self.trailing_silence = 0.0
        self.pre_roll.clear()
        return [SpeechChunk(start=start, end=end, samples=samples)]

    def _active_duration(self) -> float:
        if not self.active:
            return 0.0
        start = self.active[0][0]
        last_start, last_samples = self.active[-1]
        return last_start + last_samples.size / self.sample_rate - start

    def _trim_pre_roll(self, current_end: float) -> None:
        while self.pre_roll:
            block_start, block = self.pre_roll[0]
            block_end = block_start + block.size / self.sample_rate
            if current_end - block_end <= self.pre_roll_seconds:
                break
            self.pre_roll.popleft()


class AudioRecorder:
    def __init__(
        self,
        source: AudioSource,
        output_pcm: Path,
        on_speech_chunk: Callable[[SpeechChunk], None],
        on_error: Callable[[str], None],
        processing: AudioProcessingOptions | None = None,
    ) -> None:
        self.source = source
        self.output_pcm = Path(output_pcm)
        self.on_speech_chunk = on_speech_chunk
        self.on_error = on_error
        self.processing = processing or AudioProcessingOptions()
        self.vad = AdaptiveVad()
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=100)
        self._worker: threading.Thread | None = None
        self._pa = None
        self._stream = None
        self._file = None
        self._sample_cursor = 0
        self._paused = threading.Event()
        self._stopping = threading.Event()
        self._failed = threading.Event()
        self._vad_lock = threading.Lock()
        self._last_error_monotonic = 0.0
        # Last time the PortAudio callback delivered data. A watchdog reads
        # this to detect a silent stall (mic unplugged, sleep/resume) where
        # the callback simply stops firing without any error.
        self._last_data_monotonic = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        return self._sample_cursor / SAMPLE_RATE

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    @property
    def failed(self) -> bool:
        """True when the processing worker died (e.g. disk full)."""
        return self._failed.is_set()

    @property
    def seconds_since_last_data(self) -> float:
        """Seconds since the audio callback last delivered any frames.

        Grows without bound when the device stops calling back (mic
        unplugged, sleep/resume); the recording watchdog uses this to warn
        the student that no sound is arriving."""
        return time.monotonic() - self._last_data_monotonic

    def _report_throttled_error(self, message: str) -> None:
        """The PortAudio callback fires every ~100 ms; rate-limit its errors
        so one persistent condition cannot flood the UI event queue."""
        now = time.monotonic()
        if now - self._last_error_monotonic >= 5.0:
            self._last_error_monotonic = now
            self.on_error(message)

    def start(self) -> None:
        import pyaudiowpatch as pyaudio

        self.output_pcm.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_pcm.open("wb")
        frames_per_buffer = max(320, self.source.sample_rate // 10)

        def callback(in_data, frame_count, time_info, status_flags):
            del frame_count, time_info
            # A float store is atomic in CPython; no lock needed for the
            # watchdog that only reads this value.
            self._last_data_monotonic = time.monotonic()
            if status_flags:
                self._report_throttled_error(f"Audio input status: {status_flags}")
            if not self._stopping.is_set() and not self._failed.is_set():
                try:
                    self._queue.put_nowait(in_data)
                except queue.Full:
                    self._report_throttled_error(
                        "Audio buffer is full. A short section may be missing."
                    )
            return (None, pyaudio.paContinue)

        try:
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self.source.channels,
                rate=self.source.sample_rate,
                input=True,
                input_device_index=self.source.device_index,
                frames_per_buffer=frames_per_buffer,
                stream_callback=callback,
            )
        except Exception:
            # Release everything acquired so far: a leaked file handle keeps
            # the lesson folder undeletable on Windows.
            if self._stream is not None:
                try:
                    self._stream.close()
                except OSError:
                    pass
                self._stream = None
            if self._pa is not None:
                self._pa.terminate()
                self._pa = None
            self._file.close()
            self._file = None
            raise
        self._worker = threading.Thread(target=self._process_audio, name="audio-processor", daemon=True)
        self._worker.start()
        self._last_data_monotonic = time.monotonic()
        self._stream.start_stream()

    def toggle_pause(self) -> bool:
        if self._paused.is_set():
            self._paused.clear()
        else:
            self._paused.set()
            with self._vad_lock:
                self._emit(self.vad.flush(force=True))
        return self._paused.is_set()

    def _signal_worker_stop(self) -> None:
        """Deliver the stop sentinel even when the queue is full.

        If the worker died (disk full etc.) the queue fills within seconds
        and a blocking put would deadlock the whole stop sequence, leaving
        the recording impossible to finish."""
        while True:
            try:
                self._queue.put_nowait(None)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

    def stop(self) -> None:
        self._stopping.set()
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except OSError:
                pass
            self._stream = None
        self._signal_worker_stop()
        worker = self._worker
        self._worker = None
        if worker is not None:
            # The backlog is at most ~10 s of audio; 30 s is generous even
            # on a slow CPU. Never close the file under a live worker.
            worker.join(timeout=30)
            if worker.is_alive():
                self._failed.set()
                self.on_error("録音処理の終了待ちがタイムアウトしました。音声の末尾が欠ける可能性があります。")
                if self._pa is not None:
                    self._pa.terminate()
                    self._pa = None
                return
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except OSError:
                pass
            self._file = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def _process_audio(self) -> None:
        try:
            from scipy.signal import resample_poly

            divisor = math.gcd(self.source.sample_rate, SAMPLE_RATE)
            up = SAMPLE_RATE // divisor
            down = self.source.sample_rate // divisor
            while True:
                data = self._queue.get()
                if data is None:
                    break
                if self._paused.is_set():
                    continue
                samples = np.frombuffer(data, dtype=np.int16)
                if self.source.channels > 1:
                    samples = samples.reshape(-1, self.source.channels).mean(axis=1)
                if self.source.sample_rate != SAMPLE_RATE:
                    samples = resample_poly(samples.astype(np.float32), up, down)
                samples = np.clip(np.rint(samples), -32768, 32767).astype(np.int16)
                samples = process_audio_samples(samples, self.processing)
                start = self._sample_cursor / SAMPLE_RATE
                self._file.write(samples.tobytes())
                self._file.flush()
                self._sample_cursor += samples.size
                with self._vad_lock:
                    self._emit(self.vad.process(samples, start))
            with self._vad_lock:
                self._emit(self.vad.flush(force=True))
        except Exception as exc:
            # Fatal worker error (typically the disk is full). Mark the
            # recorder failed so the callback stops enqueueing, then keep
            # draining until the stop sentinel arrives so stop() never hangs.
            self._failed.set()
            log_exception("録音の書き込みに失敗", exc)
            self.on_error("録音を書き込めません。パソコンの空き容量を確認してください。")
            while True:
                try:
                    data = self._queue.get(timeout=1.0)
                except queue.Empty:
                    if self._stopping.is_set():
                        return
                    continue
                if data is None:
                    return

    def _emit(self, chunks: list[SpeechChunk]) -> None:
        for chunk in chunks:
            self.on_speech_chunk(chunk)


def convert_pcm_to_opus(ffmpeg: Path, pcm_path: Path, opus_path: Path) -> None:
    temporary = opus_path.with_name(f"{opus_path.stem}.tmp{opus_path.suffix}")
    command = [
        str(ffmpeg), "-y", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-i", str(pcm_path), "-c:a", "libopus", "-b:a", "40k", "-vbr", "on", str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not temporary.exists() or temporary.stat().st_size < 100:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip() or "Opus conversion failed")
    temporary.replace(opus_path)


def convert_audio_to_pcm(ffmpeg: Path, source_path: Path, pcm_path: Path) -> None:
    temporary = pcm_path.with_suffix(".pcm.tmp")
    command = [
        str(ffmpeg), "-y", "-i", str(source_path), "-f", "s16le",
        "-ar", str(SAMPLE_RATE), "-ac", "1", str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not temporary.exists() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip() or "Audio import conversion failed")
    temporary.replace(pcm_path)


class AudioPlayer:
    def __init__(
        self,
        on_position: Callable[[float], None],
        on_finished: Callable[[], None],
        on_error: Callable[[str], None],
        ffmpeg: Path,
    ) -> None:
        self.on_position = on_position
        self.on_finished = on_finished
        self.on_error = on_error
        self.ffmpeg = Path(ffmpeg)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._path: Path | None = None
        self._position = 0.0
        self._start = 0.0
        self._max_seconds = 60.0
        self._speed = 1.0

    @property
    def position(self) -> float:
        return self._position

    @property
    def playing(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def play(self, path: Path, start: float, max_seconds: float = 60.0) -> None:
        self.stop()
        self._path = Path(path)
        self._start = max(0.0, start)
        self._max_seconds = max_seconds
        self._position = self._start
        self._stop.clear()
        self._pause.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(self._path, self._start, max_seconds),
            name="audio-player",
            daemon=True,
        )
        self._thread.start()

    def toggle_pause(self) -> bool:
        if self._pause.is_set():
            self._pause.clear()
            return False
        self._pause.set()
        return True

    def back_five_seconds(self) -> None:
        if self._path is not None:
            self.play(self._path, max(0.0, self._position - 5.0), self._max_seconds)

    def forward_five_seconds(self) -> None:
        if self._path is not None:
            self.play(self._path, self._position + 5.0, self._max_seconds)

    def set_speed(self, speed: float) -> None:
        self._speed = min(2.0, max(0.5, float(speed)))
        if self._path is not None and self.playing:
            self.play(self._path, self._position, self._max_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)
        self._thread = None

    def _run(self, path: Path, start: float, max_seconds: float) -> None:
        process: subprocess.Popen | None = None
        stream = None
        try:
            import sounddevice as sd

            input_args = (
                ["-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1"]
                if path.suffix.lower() == ".pcm"
                else []
            )
            process = subprocess.Popen(
                [
                    str(self.ffmpeg), "-v", "error", *input_args,
                    "-ss", f"{start:.3f}", "-i", str(path),
                    "-t", f"{max_seconds:.3f}", "-af", f"atempo={self._speed:.2f}",
                    "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            reader = process.stdout
            stream = sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
            stream.start()
            frames_played = 0
            block_bytes = 3200
            while not self._stop.is_set() and frames_played / SAMPLE_RATE * self._speed < max_seconds:
                if self._pause.is_set():
                    time.sleep(0.05)
                    continue
                data = reader.read(block_bytes)
                if not data:
                    break
                stream.write(data)
                frames_played += len(data) // 2
                self._position = start + frames_played / SAMPLE_RATE * self._speed
                self.on_position(self._position)
            reader.close()
        except Exception as exc:
            log_exception("音声の再生に失敗", exc)
            self.on_error("音声を再生できませんでした。もう一度お試しください。")
        finally:
            if stream is not None:
                stream.stop()
                stream.close()
            if process is not None and process.poll() is None:
                process.terminate()
            self.on_finished()
