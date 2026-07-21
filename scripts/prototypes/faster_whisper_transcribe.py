"""faster-whisper transcription test with built-in Silero VAD."""

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="faster-whisper CPU/int8 transcription test")
    parser.add_argument("--input",      required=True,  help="Input WAV file")
    parser.add_argument("--output_dir", required=True,  help="Output directory")
    parser.add_argument("--log",        required=True,  help="Log file path")
    parser.add_argument("--model",      default="small", help="Model size: tiny/base/small/medium/large-v3")
    parser.add_argument("--language",   default=None,   help="Language code (e.g. ja, en) or None for auto-detect")
    parser.add_argument("--compute_type", default="int8", help="Quantization: int8 / float16 / float32")
    parser.add_argument("--vad_filter", action="store_true", default=True, help="Enable built-in Silero VAD")
    parser.add_argument("--model_dir",  default=None,   help="Local model cache dir (None = HuggingFace default)")
    return parser.parse_args()


def append_log(log_path: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{stamp} {message}\n")


def main() -> int:
    args = parse_args()
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    append_log(log_path, "faster_whisper_start")
    append_log(log_path, f"python={sys.version.replace(chr(10), ' ')}")
    append_log(log_path, f"platform={platform.platform()}")
    append_log(log_path, f"model={args.model} compute_type={args.compute_type} language={args.language} vad_filter={args.vad_filter}")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        append_log(log_path, "ERROR import faster_whisper failed")
        raise

    print(f"Loading faster-whisper model: {args.model} / cpu / {args.compute_type}", flush=True)
    model_load_start = time.perf_counter()
    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type=args.compute_type,
        download_root=args.model_dir,
    )
    model_load_seconds = time.perf_counter() - model_load_start
    append_log(log_path, f"model_load_seconds={model_load_seconds:.3f}")
    print(f"Model loaded in {model_load_seconds:.3f}s", flush=True)

    print(f"Transcribing: {args.input}", flush=True)
    asr_start = time.perf_counter()

    segments, info = model.transcribe(
        args.input,
        language=args.language if args.language and args.language != "None" else None,
        vad_filter=args.vad_filter,
        beam_size=5,
    )

    lines = []
    for seg in segments:
        line = f"[{seg.start:.2f} --> {seg.end:.2f}]  {seg.text.strip()}"
        lines.append(line)
        print(line, flush=True)

    asr_seconds = time.perf_counter() - asr_start
    append_log(log_path, f"detected_language={info.language} (p={info.language_probability:.3f})")
    append_log(log_path, f"asr_seconds={asr_seconds:.3f}")
    append_log(log_path, f"segments={len(lines)}")

    transcript_path = output_dir / "full_transcript.txt"
    transcript_path.write_text("\n".join(lines), encoding="utf-8")

    total_seconds = time.perf_counter() - started
    summary = {
        "model": args.model,
        "compute_type": args.compute_type,
        "language_hint": args.language,
        "detected_language": info.language,
        "detected_language_probability": round(info.language_probability, 4),
        "vad_filter": args.vad_filter,
        "model_load_seconds": round(model_load_seconds, 3),
        "asr_seconds": round(asr_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "segments": len(lines),
        "full_transcript": str(transcript_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_log(log_path, f"total_seconds={total_seconds:.3f}")
    append_log(log_path, f"full_transcript={transcript_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
