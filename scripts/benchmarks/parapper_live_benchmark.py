"""Play a WAV file and measure a running Parapper process."""

from __future__ import annotations

import argparse
import json
import threading
import time
import wave
from pathlib import Path

import numpy as np
import psutil
import sounddevice as sd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--output-device", type=int)
    return parser.parse_args()


def load_wav(path: Path, seconds: float | None) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
        if seconds is not None:
            frame_count = min(frame_count, round(seconds * sample_rate))
        raw = source.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported, got {sample_width * 8}-bit")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return audio.reshape(-1, channels), sample_rate


def main() -> None:
    args = parse_args()
    audio, sample_rate = load_wav(args.audio, args.seconds)
    process = psutil.Process(args.pid)
    samples: list[dict[str, float]] = []
    stop_event = threading.Event()

    def monitor() -> None:
        while not stop_event.wait(0.1):
            try:
                memory = process.memory_info()
                cpu = process.cpu_times()
            except psutil.Error:
                break
            samples.append(
                {
                    "monotonic": time.monotonic(),
                    "rss": float(memory.rss),
                    "private": float(getattr(memory, "private", 0)),
                    "cpu": float(cpu.user + cpu.system),
                }
            )

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    start_wall = time.time()
    start_monotonic = time.monotonic()
    start_cpu = sum(process.cpu_times()[:2])
    sd.play(audio, sample_rate, device=args.output_device, blocking=True)
    playback_end = time.monotonic()
    time.sleep(3.0)
    stop_event.set()
    monitor_thread.join()
    end_cpu = sum(process.cpu_times()[:2])

    result = {
        "audio": str(args.audio.resolve()),
        "sample_rate": sample_rate,
        "channels": int(audio.shape[1]),
        "audio_seconds": len(audio) / sample_rate,
        "start_timestamp": start_wall,
        "start_monotonic": start_monotonic,
        "playback_end_monotonic": playback_end,
        "measurement_end_monotonic": time.monotonic(),
        "parapper_pid": args.pid,
        "cpu_seconds": end_cpu - start_cpu,
        "peak_rss_bytes": max((sample["rss"] for sample in samples), default=0),
        "peak_private_bytes": max((sample["private"] for sample in samples), default=0),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}))


if __name__ == "__main__":
    main()
