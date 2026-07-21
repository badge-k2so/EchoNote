"""Patch likely English/mixed chunks after a fast Japanese ReazonSpeech pass."""

import argparse
import json
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid Japanese-first ASR with English patch engine")
    parser.add_argument("--reazon_run_dir", required=True, help="Existing Reazon VAD run directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for hybrid files")
    parser.add_argument("--log", required=True, help="Log path")
    parser.add_argument("--patch_engine", choices=("fw", "parakeet"), default="fw")
    parser.add_argument("--model", default="small", help="faster-whisper model")
    parser.add_argument("--compute_type", default="int8", help="faster-whisper compute type")
    parser.add_argument("--language", default="None", help="None for auto, or en/ja/etc.")
    parser.add_argument("--parakeet_model_dir", default="models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8")
    parser.add_argument("--num_threads", type=int, default=2)
    parser.add_argument("--short_text_chars", type=int, default=8)
    parser.add_argument("--min_candidate_seconds", type=float, default=4.0)
    parser.add_argument("--max_patch_chunks", type=int, default=8)
    return parser.parse_args()


def append_log(log_path: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{stamp} {message}\n")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def compact(text: str) -> str:
    text = re.sub(r"^={3,}.*={3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", "", text)
    return text


def has_latin_word(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}", text))


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


def suspicious_short_patch(text: str) -> bool:
    norm = re.sub(r"[^a-z ]+", "", text.lower()).strip()
    norm = re.sub(r"\s+", " ", norm)
    if not norm:
        return True
    suspicious = {
        "okay",
        "ok",
        "yeah",
        "yes",
        "no",
        "thank you",
        "thanks",
        "thats not so",
        "thats it",
        "i dont know",
        "what was that",
    }
    if norm in suspicious:
        return True
    words = norm.split()
    repeated = len(words) >= 4 and len(set(words)) <= 2
    return repeated


def patch_is_usable(text: str) -> tuple[bool, str]:
    if not text.strip():
        return False, "empty_patch"
    if not has_latin_word(text):
        return False, "no_latin_word"
    if suspicious_short_patch(text):
        return False, "suspicious_short_patch"
    return True, ""


def format_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins:02d}:{secs:05.2f}"


def load_parakeet(model_dir: Path, num_threads: int):
    import sherpa_onnx

    encoder = model_dir / "encoder.int8.onnx"
    decoder = model_dir / "decoder.int8.onnx"
    joiner = model_dir / "joiner.int8.onnx"
    tokens = model_dir / "tokens.txt"
    for path in (encoder, decoder, joiner, tokens):
        if not path.is_file():
            raise FileNotFoundError(f"Parakeet model file not found: {path}")

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(encoder),
        decoder=str(decoder),
        joiner=str(joiner),
        tokens=str(tokens),
        num_threads=num_threads,
        decoding_method="greedy_search",
        provider="cpu",
        model_type="nemo_transducer",
    )


def transcribe_parakeet(recognizer, chunk_path: Path, start_offset: float) -> tuple[str, str, dict]:
    samples, sample_rate = read_wav_mono_float32(chunk_path)
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    text = getattr(stream.result, "text", "").strip()
    if not text:
        return "", "", {"detected_language": "en", "detected_language_probability": None}
    line = f"[{format_time(start_offset)} --> {format_time(start_offset + len(samples) / sample_rate)}]  {text}"
    return line, text, {"detected_language": "en", "detected_language_probability": None}


def main() -> int:
    args = parse_args()
    reazon_run = Path(args.reazon_run_dir)
    output_dir = Path(args.output_dir)
    log_path = Path(args.log)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    chunks_meta_path = reazon_run / "chunks_vad" / "vad_splits.json"
    reazon_output_dir = reazon_run / "output"
    if not chunks_meta_path.exists():
        raise FileNotFoundError(f"vad_splits.json not found: {chunks_meta_path}")

    meta = json.loads(chunks_meta_path.read_text(encoding="utf-8"))
    chunks = meta["chunks"]
    started = time.perf_counter()
    append_log(log_path, "hybrid_english_patch_start")
    append_log(log_path, f"reazon_run_dir={reazon_run}")
    append_log(log_path, f"chunks={len(chunks)}")
    append_log(log_path, f"patch_engine={args.patch_engine}")

    candidates = []
    rows = []
    for chunk in chunks:
        index = int(chunk["index"])
        duration = float(chunk["duration_seconds"])
        reazon_text = read_text(reazon_output_dir / f"chunk_{index:03d}.txt")
        norm = compact(reazon_text)
        chars = len(norm)
        score = duration / max(chars, 1)
        reason = ""
        if duration >= args.min_candidate_seconds and chars <= args.short_text_chars:
            reason = "short_reazon_output"
            candidates.append((score, chunk, reazon_text, reason))
        rows.append(
            {
                "index": index,
                "start_seconds": float(chunk["start_seconds"]),
                "duration_seconds": duration,
                "reazon_chars": chars,
                "candidate": bool(reason),
                "candidate_reason": reason,
                "score": round(score, 3),
            }
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[: max(args.max_patch_chunks, 0)]
    selected_indexes = {int(item[1]["index"]) for item in selected}
    append_log(log_path, f"candidates={len(candidates)} selected={len(selected)}")

    patch_results = {}
    model_load_seconds = 0.0
    patch_asr_seconds = 0.0

    if selected:
        load_start = time.perf_counter()
        if args.patch_engine == "fw":
            from faster_whisper import WhisperModel

            print(
                f"Loading faster-whisper once for patch chunks: {args.model} / cpu / {args.compute_type}",
                flush=True,
            )
            patch_model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)
        else:
            model_dir = Path(args.parakeet_model_dir)
            if not model_dir.is_absolute():
                model_dir = Path.cwd() / model_dir
            print(f"Loading Parakeet once for patch chunks: {model_dir}", flush=True)
            patch_model = load_parakeet(model_dir, args.num_threads)
        model_load_seconds = time.perf_counter() - load_start
        append_log(log_path, f"patch_model_load_seconds={model_load_seconds:.3f}")

        patches_dir = output_dir / "patch_chunks"
        patches_dir.mkdir(parents=True, exist_ok=True)

        for n, (_score, chunk, reazon_text, reason) in enumerate(selected, start=1):
            index = int(chunk["index"])
            start_offset = float(chunk["start_seconds"])
            chunk_path = Path(chunk["file"])
            print(f"[{n}/{len(selected)}] patch chunk_{index:03d}.wav", flush=True)
            asr_start = time.perf_counter()
            if args.patch_engine == "fw":
                segments, info = patch_model.transcribe(
                    str(chunk_path),
                    language=args.language if args.language and args.language != "None" else None,
                    vad_filter=True,
                    beam_size=5,
                )
                lines = []
                plain = []
                metadata = {
                    "detected_language": info.language,
                    "detected_language_probability": round(info.language_probability, 4),
                }
                for seg in segments:
                    start = start_offset + float(seg.start)
                    end = start_offset + float(seg.end)
                    text = seg.text.strip()
                    if not text:
                        continue
                    lines.append(f"[{format_time(start)} --> {format_time(end)}]  {text}")
                    plain.append(text)
                patch_text = "\n".join(lines).strip()
                patch_plain = " ".join(plain).strip()
            else:
                patch_text, patch_plain, metadata = transcribe_parakeet(patch_model, chunk_path, start_offset)
            seconds = time.perf_counter() - asr_start
            patch_asr_seconds += seconds
            usable, reject_reason = patch_is_usable(patch_plain)
            (patches_dir / f"patch_{index:03d}.txt").write_text(patch_text, encoding="utf-8")
            patch_results[index] = {
                "index": index,
                "reason": reason,
                "seconds": round(seconds, 3),
                "patch_engine": args.patch_engine,
                "detected_language": metadata["detected_language"],
                "detected_language_probability": metadata["detected_language_probability"],
                "reazon_text": reazon_text,
                "patch_text": patch_text,
                "patch_plain": patch_plain,
                "patch_has_latin_word": has_latin_word(patch_plain),
                "patch_usable": usable,
                "patch_reject_reason": reject_reason,
            }

    review_lines = []
    best_effort_lines = []
    for chunk in chunks:
        index = int(chunk["index"])
        start = float(chunk["start_seconds"])
        duration = float(chunk["duration_seconds"])
        end = start + duration
        reazon_text = read_text(reazon_output_dir / f"chunk_{index:03d}.txt")
        patch = patch_results.get(index)
        header = f"===== chunk_{index:03d} {format_time(start)}-{format_time(end)} ====="
        review_lines.append(header)
        review_lines.append("[ja_base]")
        review_lines.append(reazon_text if reazon_text else "(empty)")
        if patch:
            review_lines.append(f"[{args.patch_engine}_patch]")
            review_lines.append(patch["patch_text"] if patch["patch_text"] else "(empty)")
            if not patch["patch_usable"]:
                review_lines.append(f"[patch_rejected] {patch['patch_reject_reason']}")
        review_lines.append("")

        if patch and patch["patch_usable"]:
            best_effort_lines.append(header)
            best_effort_lines.append(patch["patch_text"])
        else:
            best_effort_lines.append(header)
            best_effort_lines.append(reazon_text)
        best_effort_lines.append("")

    review_path = output_dir / "hybrid_review_transcript.txt"
    best_path = output_dir / "hybrid_best_effort.txt"
    candidates_path = output_dir / "hybrid_candidates.json"
    summary_path = output_dir / "hybrid_summary.json"
    review_path.write_text("\n".join(review_lines).strip() + "\n", encoding="utf-8")
    best_path.write_text("\n".join(best_effort_lines).strip() + "\n", encoding="utf-8")
    candidates_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    total_seconds = time.perf_counter() - started
    summary = {
        "mode": f"ja_first_{args.patch_engine}_patch",
        "reazon_run_dir": str(reazon_run),
        "chunks": len(chunks),
        "candidates": len(candidates),
        "selected_patch_chunks": len(selected),
        "selected_indexes": sorted(selected_indexes),
        "patch_engine": args.patch_engine,
        "faster_whisper_model": args.model if args.patch_engine == "fw" else "",
        "parakeet_model_dir": args.parakeet_model_dir if args.patch_engine == "parakeet" else "",
        "compute_type": args.compute_type,
        "language": args.language,
        "model_load_seconds": round(model_load_seconds, 3),
        "patch_asr_seconds": round(patch_asr_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "review_transcript": str(review_path),
        "best_effort": str(best_path),
        "patch_results": list(patch_results.values()),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    append_log(log_path, f"total_seconds={total_seconds:.3f}")
    append_log(log_path, "hybrid_english_patch_done")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
