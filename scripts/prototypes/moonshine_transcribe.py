import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path


def append_log(log_path: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{stamp} {message}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Moonshine Voice Japanese batch transcription test")
    parser.add_argument("--chunks_dir", required=True, help="Directory containing WAV chunks")
    parser.add_argument("--output_dir", required=True, help="Directory for transcription outputs")
    parser.add_argument("--log", required=True, help="Log file path")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--max_tokens_per_second", default="13.0")
    return parser.parse_args()


def transcript_to_text(transcript) -> str:
    lines = getattr(transcript, "lines", [])
    return "\n".join(line.text for line in lines if getattr(line, "text", "").strip()).strip()


def main() -> int:
    args = parse_args()
    chunks_dir = Path(args.chunks_dir)
    output_dir = Path(args.output_dir)
    log_path = Path(args.log)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    append_log(log_path, "moonshine_python_start")
    append_log(log_path, f"python={sys.version.replace(chr(10), ' ')}")
    append_log(log_path, f"platform={platform.platform()}")
    append_log(log_path, f"language={args.language}")

    chunks = sorted(chunks_dir.glob("*.wav"))
    if not chunks:
        raise FileNotFoundError(f"No WAV chunks found in {chunks_dir}")

    try:
        import moonshine_voice
        from moonshine_voice import Transcriber
    except Exception:
        append_log(log_path, "ERROR import_failed")
        append_log(log_path, traceback.format_exc())
        raise

    model_start = time.perf_counter()
    try:
        model_path, model_arch = moonshine_voice.get_model_for_language(args.language)
        options = {"max_tokens_per_second": args.max_tokens_per_second}
        print(f"Loading Moonshine model once: path={model_path} arch={model_arch}", flush=True)
        transcriber = Transcriber(str(model_path), model_arch=model_arch, options=options)
    except Exception:
        append_log(log_path, "ERROR model_load_failed")
        append_log(log_path, traceback.format_exc())
        raise
    model_load_seconds = time.perf_counter() - model_start
    append_log(log_path, f"moonshine_model_load_seconds={model_load_seconds:.3f}")

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
            audio_data, sample_rate = moonshine_voice.load_wav_file(chunk_path)
            transcript = transcriber.transcribe_without_streaming(audio_data, sample_rate)
            text = transcript_to_text(transcript)
            output_path.write_text(text, encoding="utf-8")
            chunk_seconds = time.perf_counter() - chunk_start
            asr_total_seconds += chunk_seconds
            success += 1
            append_log(log_path, f"chunk_done index={index} seconds={chunk_seconds:.3f} chars={len(text)}")
            chunk_results.append(
                {
                    "index": index,
                    "file": str(chunk_path),
                    "output": str(output_path),
                    "success": True,
                    "seconds": chunk_seconds,
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
                    "seconds": chunk_seconds,
                    "chars": 0,
                    "error": str(exc),
                }
            )

    full_transcript_path = output_dir / "full_transcript.txt"
    with full_transcript_path.open("w", encoding="utf-8") as full:
        for chunk_path in chunks:
            full.write(f"===== {chunk_path.stem} =====\n")
            chunk_txt = output_dir / f"{chunk_path.stem}.txt"
            if chunk_txt.exists():
                full.write(chunk_txt.read_text(encoding="utf-8").strip())
            else:
                full.write("[ERROR]")
            full.write("\n\n")

    try:
        transcriber.close()
    except Exception:
        pass

    total_seconds = time.perf_counter() - started
    summary = {
        "engine": "moonshine-voice",
        "language": args.language,
        "chunks": len(chunks),
        "success": success,
        "failed": failed,
        "model_load_seconds": round(model_load_seconds, 3),
        "asr_total_seconds": round(asr_total_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "full_transcript": str(full_transcript_path),
        "chunk_results": chunk_results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_log(log_path, f"moonshine_chunks={len(chunks)}")
    append_log(log_path, f"moonshine_success={success}")
    append_log(log_path, f"moonshine_failed={failed}")
    append_log(log_path, f"moonshine_asr_total_seconds={asr_total_seconds:.3f}")
    append_log(log_path, f"moonshine_total_seconds={total_seconds:.3f}")
    append_log(log_path, f"moonshine_full_transcript={full_transcript_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
