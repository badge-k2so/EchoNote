"""sherpa-onnx Whisper offline transcription for VAD chunks."""

import argparse
import json
import platform
import sys
import time
import traceback
import wave
from pathlib import Path

import numpy as np


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def append_log(log_path: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{stamp} {message}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="sherpa-onnx Whisper CPU transcription")
    parser.add_argument("--chunks_dir", required=True, help="Directory containing WAV chunks")
    parser.add_argument("--output_dir", required=True, help="Directory for outputs")
    parser.add_argument("--log", required=True, help="Log file path")
    parser.add_argument("--encoder", required=True, help="Whisper encoder ONNX path")
    parser.add_argument("--decoder", required=True, help="Whisper decoder ONNX path")
    parser.add_argument("--tokens", required=True, help="Whisper tokens path")
    parser.add_argument("--language", default="en")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--num_threads", type=int, default=2)
    parser.add_argument("--provider", default="cpu")
    return parser.parse_args()


def read_wav_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        channels = f.getnchannels()
        sample_width = f.getsampwidth()
        sample_rate = f.getframerate()
        frames = f.readframes(f.getnframes())

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample_width={sample_width}: {path}")

    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return samples.astype(np.float32) / 32768.0, sample_rate


def result_text(result: object) -> str:
    text = getattr(result, "text", "")
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = sorted(Path(args.chunks_dir).glob("*.wav"))
    if not chunks:
        raise FileNotFoundError(f"No WAV chunks found in {args.chunks_dir}")

    started = time.perf_counter()
    append_log(log_path, "sherpa_onnx_whisper_start")
    append_log(log_path, f"python={sys.version.replace(chr(10), ' ')}")
    append_log(log_path, f"platform={platform.platform()}")
    append_log(log_path, f"encoder={args.encoder}")
    append_log(log_path, f"decoder={args.decoder}")
    append_log(log_path, f"tokens={args.tokens}")
    append_log(log_path, f"language={args.language} task={args.task} num_threads={args.num_threads} provider={args.provider}")

    try:
        import sherpa_onnx
    except Exception:
        append_log(log_path, "ERROR import_failed")
        append_log(log_path, traceback.format_exc())
        raise

    model_load_start = time.perf_counter()
    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=args.encoder,
        decoder=args.decoder,
        tokens=args.tokens,
        language=args.language,
        task=args.task,
        num_threads=args.num_threads,
        provider=args.provider,
    )
    model_load_seconds = time.perf_counter() - model_load_start
    append_log(log_path, f"model_load_seconds={model_load_seconds:.3f}")

    success = 0
    failed = 0
    asr_total_seconds = 0.0
    chunk_results = []

    for index, chunk_path in enumerate(chunks):
        output_path = output_dir / f"{chunk_path.stem}.txt"
        error_path = output_dir / f"{chunk_path.stem}.error.txt"
        print(f"[{index + 1}/{len(chunks)}] {chunk_path.name}", flush=True)
        append_log(log_path, f"chunk_start index={index} file={chunk_path}")

        chunk_start = time.perf_counter()
        try:
            samples, sample_rate = read_wav_mono_float32(chunk_path)
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            recognizer.decode_stream(stream)
            text = result_text(stream.result)
            output_path.write_text(text, encoding="utf-8")

            chunk_seconds = time.perf_counter() - chunk_start
            asr_total_seconds += chunk_seconds
            success += 1
            append_log(log_path, f"chunk_done index={index} seconds={chunk_seconds:.3f} chars={len(text)}")
            print(text, flush=True)
            chunk_results.append(
                {
                    "index": index,
                    "file": str(chunk_path),
                    "output": str(output_path),
                    "success": True,
                    "seconds": round(chunk_seconds, 3),
                    "chars": len(text),
                }
            )
        except Exception as exc:
            chunk_seconds = time.perf_counter() - chunk_start
            failed += 1
            error_text = traceback.format_exc()
            error_path.write_text(error_text, encoding="utf-8")
            append_log(log_path, f"ERROR chunk_failed index={index} seconds={chunk_seconds:.3f} error={exc}")
            append_log(log_path, error_text)
            chunk_results.append(
                {
                    "index": index,
                    "file": str(chunk_path),
                    "output": str(error_path),
                    "success": False,
                    "seconds": round(chunk_seconds, 3),
                    "chars": 0,
                    "error": str(exc),
                }
            )

    full_transcript_path = output_dir / "full_transcript.txt"
    with full_transcript_path.open("w", encoding="utf-8") as full:
        for chunk_path in chunks:
            chunk_txt = output_dir / f"{chunk_path.stem}.txt"
            full.write(f"===== {chunk_path.stem} =====\n")
            full.write(chunk_txt.read_text(encoding="utf-8").strip() if chunk_txt.exists() else "[ERROR]")
            full.write("\n\n")

    total_seconds = time.perf_counter() - started
    summary = {
        "engine": "sherpa-onnx-whisper",
        "encoder": args.encoder,
        "decoder": args.decoder,
        "tokens": args.tokens,
        "language": args.language,
        "task": args.task,
        "num_threads": args.num_threads,
        "provider": args.provider,
        "chunks": len(chunks),
        "success": success,
        "failed": failed,
        "model_load_seconds": round(model_load_seconds, 3),
        "asr_total_seconds": round(asr_total_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "full_transcript": str(full_transcript_path),
        "chunk_results": chunk_results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_log(log_path, f"sherpa_chunks={len(chunks)}")
    append_log(log_path, f"sherpa_success={success}")
    append_log(log_path, f"sherpa_failed={failed}")
    append_log(log_path, f"sherpa_asr_total_seconds={asr_total_seconds:.3f}")
    append_log(log_path, f"sherpa_total_seconds={total_seconds:.3f}")
    append_log(log_path, f"sherpa_full_transcript={full_transcript_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
