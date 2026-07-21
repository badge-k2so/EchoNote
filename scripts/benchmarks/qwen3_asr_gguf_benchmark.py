from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mmproj", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input", action="append", required=True, help="LABEL=CHUNKS_DIR")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx_size", type=int, default=8192)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--port", type=int, default=18080)
    return parser.parse_args()


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(url: str, process: subprocess.Popen, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited during startup: {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.25)
    raise TimeoutError("Timed out waiting for llama-server")


def parse_asr_content(content: str) -> tuple[str, str]:
    language_match = re.search(r"language\s+([^<\r\n]+)", content, flags=re.IGNORECASE)
    language = language_match.group(1).strip() if language_match else "unknown"
    text_match = re.search(r"<asr_text>(.*)", content, flags=re.DOTALL | re.IGNORECASE)
    transcript = text_match.group(1).strip() if text_match else content.strip()
    return language, transcript


def monitor_memory(process: subprocess.Popen, state: dict, stop: threading.Event) -> None:
    tracked = psutil.Process(process.pid)
    while not stop.wait(0.1):
        try:
            rss = tracked.memory_info().rss
            for child in tracked.children(recursive=True):
                rss += child.memory_info().rss
            state["peak_rss"] = max(state["peak_rss"], rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_responses"
    raw_dir.mkdir(exist_ok=True)

    inputs: list[tuple[str, Path]] = []
    for value in args.input:
        if "=" not in value:
            raise ValueError(f"Invalid --input value: {value}")
        label, directory = value.split("=", 1)
        inputs.append((label, Path(directory)))

    stdout_path = output_dir / "llama_server_stdout.log"
    stderr_path = output_dir / "llama_server_stderr.log"
    command = [
        args.server,
        "-m", args.model,
        "-mm", args.mmproj,
        "-c", str(args.ctx_size),
        "-t", str(args.threads),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--no-webui",
        "--no-warmup",
    ]

    creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    start = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        process = subprocess.Popen(
            command,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            creationflags=creationflags,
        )
        memory_state = {"peak_rss": 0}
        stop_monitor = threading.Event()
        monitor = threading.Thread(
            target=monitor_memory,
            args=(process, memory_state, stop_monitor),
            daemon=True,
        )
        monitor.start()
        try:
            wait_for_server(f"http://127.0.0.1:{args.port}/health", process)
            load_seconds = time.monotonic() - start
            results: list[dict] = []

            for label, chunks_dir in inputs:
                chunks = sorted(chunks_dir.glob("*.wav"))
                if not chunks:
                    raise FileNotFoundError(f"No WAV chunks found: {chunks_dir}")
                label_start = time.monotonic()
                chunk_results: list[dict] = []
                transcript_blocks: list[str] = []
                audio_seconds = 0.0

                for index, chunk in enumerate(chunks):
                    duration = wav_seconds(chunk)
                    audio_seconds += duration
                    audio_b64 = base64.b64encode(chunk.read_bytes()).decode("ascii")
                    payload = {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_audio",
                                        "input_audio": {"data": audio_b64, "format": "wav"},
                                    },
                                    {
                                        "type": "text",
                                        "text": "Transcribe the audio faithfully. Output only the transcript.",
                                    },
                                ],
                            }
                        ],
                        "temperature": 0.0,
                        "max_tokens": args.max_tokens,
                        "stream": False,
                    }
                    chunk_start = time.monotonic()
                    response = post_json(
                        f"http://127.0.0.1:{args.port}/v1/chat/completions",
                        payload,
                        timeout=600,
                    )
                    seconds = time.monotonic() - chunk_start
                    content = response["choices"][0]["message"]["content"]
                    language, transcript = parse_asr_content(content)
                    (raw_dir / f"{label}_{index:03d}.json").write_text(
                        json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    transcript_blocks.append(
                        f"===== chunk_{index:03d} language={language} =====\n{transcript}"
                    )
                    chunk_results.append(
                        {
                            "index": index,
                            "file": str(chunk),
                            "audio_seconds": round(duration, 3),
                            "inference_seconds": round(seconds, 3),
                            "language": language,
                            "text": transcript,
                            "timings": response.get("timings", {}),
                        }
                    )
                    print(
                        f"[{label} {index + 1}/{len(chunks)}] "
                        f"audio={duration:.1f}s infer={seconds:.1f}s lang={language}",
                        flush=True,
                    )

                inference_seconds = time.monotonic() - label_start
                (output_dir / f"{label}_transcript.txt").write_text(
                    "\n\n".join(transcript_blocks) + "\n", encoding="utf-8"
                )
                results.append(
                    {
                        "label": label,
                        "chunks_dir": str(chunks_dir),
                        "chunks": len(chunks),
                        "audio_seconds": round(audio_seconds, 3),
                        "inference_seconds": round(inference_seconds, 3),
                        "real_time_factor": round(inference_seconds / audio_seconds, 4),
                        "chunk_results": chunk_results,
                    }
                )
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            stop_monitor.set()
            monitor.join(timeout=2)

    summary = {
        "engine": args.server,
        "model": args.model,
        "mmproj": args.mmproj,
        "threads": args.threads,
        "ctx_size": args.ctx_size,
        "max_tokens": args.max_tokens,
        "model_load_seconds": round(load_seconds, 3),
        "peak_rss_mb": round(memory_state["peak_rss"] / (1024 * 1024), 1),
        "total_wall_seconds": round(time.monotonic() - start, 3),
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
