"""Japanese/English alternating classroom ASR patcher.

This mode keeps ReazonSpeech as the fast Japanese pass, but patches many
likely-English chunks with faster-whisper and writes language hints for the
post-processing step.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


FILLER_JA = {"", "あ", "ああ", "あっ", "うん", "え", "はい", "いや", "そう", "フフ"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Japanese/English alternating ASR mode")
    parser.add_argument("--reazon_run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--model", default="small")
    parser.add_argument("--compute_type", default="int8")
    parser.add_argument("--language", default="en")
    parser.add_argument("--short_text_chars", type=int, default=16)
    parser.add_argument("--min_candidate_seconds", type=float, default=2.0)
    parser.add_argument("--max_patch_chunks", type=int, default=240)
    parser.add_argument("--pair_gap_seconds", type=float, default=45.0)
    return parser.parse_args()


def append_log(log_path: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{stamp} {message}\n")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def compact(text: str) -> str:
    text = re.sub(r"^={3,}.*={3,}\s*$", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", "", text)


def has_latin_word(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}", text))


def has_japanese(text: str) -> bool:
    return bool(re.search(r"[ぁ-んァ-ン一-龥]", text))


def format_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins:02d}:{secs:05.2f}"


def classify_chunk(reazon_text: str, patch_plain: str | None) -> str:
    reazon_norm = compact(reazon_text)
    patch = patch_plain or ""
    if patch and has_latin_word(patch):
        if reazon_norm and reazon_norm not in FILLER_JA and len(reazon_norm) > 12:
            return "mixed"
        return "en"
    if reazon_norm and reazon_norm not in FILLER_JA:
        return "ja"
    return "unknown"


def is_patch_candidate(reazon_text: str, duration: float, short_chars: int, min_seconds: float) -> bool:
    norm = compact(reazon_text)
    if duration < min_seconds:
        return False
    if norm in FILLER_JA:
        return True
    return len(norm) <= short_chars


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

    chunks = json.loads(chunks_meta_path.read_text(encoding="utf-8"))["chunks"]
    append_log(log_path, "ja_en_alternating_patch_start")
    append_log(log_path, f"reazon_run_dir={reazon_run}")
    append_log(log_path, f"chunks={len(chunks)}")

    candidates = []
    rows = []
    for chunk in chunks:
        index = int(chunk["index"])
        duration = float(chunk["duration_seconds"])
        reazon_text = read_text(reazon_output_dir / f"chunk_{index:03d}.txt")
        norm = compact(reazon_text)
        candidate = is_patch_candidate(reazon_text, duration, args.short_text_chars, args.min_candidate_seconds)
        score = duration / max(len(norm), 1)
        if candidate:
            candidates.append((score, chunk, reazon_text))
        rows.append({
            "index": index,
            "start_seconds": float(chunk["start_seconds"]),
            "duration_seconds": duration,
            "reazon_chars": len(norm),
            "candidate": candidate,
            "score": round(score, 3),
        })

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[: max(args.max_patch_chunks, 0)]
    selected_indexes = {int(item[1]["index"]) for item in selected}
    append_log(log_path, f"candidates={len(candidates)} selected={len(selected)}")

    patch_results = {}
    model_load_seconds = 0.0
    patch_asr_seconds = 0.0
    if selected:
        from faster_whisper import WhisperModel

        print(f"Loading faster-whisper once: {args.model} / cpu / {args.compute_type} / {args.language}", flush=True)
        load_start = time.perf_counter()
        model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)
        model_load_seconds = time.perf_counter() - load_start
        append_log(log_path, f"patch_model_load_seconds={model_load_seconds:.3f}")

        patches_dir = output_dir / "patch_chunks"
        patches_dir.mkdir(parents=True, exist_ok=True)
        for n, (_score, chunk, reazon_text) in enumerate(selected, start=1):
            index = int(chunk["index"])
            start_offset = float(chunk["start_seconds"])
            chunk_path = Path(chunk["file"])
            print(f"[{n}/{len(selected)}] patch chunk_{index:03d}.wav", flush=True)
            asr_start = time.perf_counter()
            segments, info = model.transcribe(
                str(chunk_path),
                language=args.language if args.language and args.language != "None" else None,
                vad_filter=True,
                beam_size=5,
            )
            lines = []
            plain = []
            for seg in segments:
                start = start_offset + float(seg.start)
                end = start_offset + float(seg.end)
                text = seg.text.strip()
                if not text:
                    continue
                lines.append(f"[{format_time(start)} --> {format_time(end)}]  {text}")
                plain.append(text)
            seconds = time.perf_counter() - asr_start
            patch_asr_seconds += seconds
            patch_text = "\n".join(lines).strip()
            patch_plain = " ".join(plain).strip()
            (patches_dir / f"patch_{index:03d}.txt").write_text(patch_text, encoding="utf-8")
            patch_results[index] = {
                "index": index,
                "seconds": round(seconds, 3),
                "detected_language": info.language,
                "detected_language_probability": round(info.language_probability, 4),
                "reazon_text": reazon_text,
                "patch_text": patch_text,
                "patch_plain": patch_plain,
                "patch_has_latin_word": has_latin_word(patch_plain),
            }

    review_lines = []
    best_effort_lines = []
    timeline = []
    for chunk in chunks:
        index = int(chunk["index"])
        start = float(chunk["start_seconds"])
        duration = float(chunk["duration_seconds"])
        end = start + duration
        reazon_text = read_text(reazon_output_dir / f"chunk_{index:03d}.txt")
        patch = patch_results.get(index)
        patch_plain = patch["patch_plain"] if patch else ""
        lang_hint = classify_chunk(reazon_text, patch_plain)
        header = f"===== chunk_{index:03d} {format_time(start)}-{format_time(end)} ====="
        review_lines.extend([header, f"[lang_hint] {lang_hint}", "[ja_base]", reazon_text if reazon_text else "(empty)"])
        if patch:
            review_lines.extend(["[fw_patch]", patch["patch_text"] if patch["patch_text"] else "(empty)"])
        review_lines.append("")

        best_effort_lines.append(header)
        if lang_hint in {"en", "mixed"} and patch and patch["patch_text"]:
            best_effort_lines.append(patch["patch_text"])
        else:
            best_effort_lines.append(reazon_text)
        best_effort_lines.append("")

        timeline.append({
            "index": index,
            "start_seconds": start,
            "end_seconds": end,
            "lang_hint": lang_hint,
            "ja_base": reazon_text,
            "fw_patch": patch_plain,
        })

    pairs = []
    for i, row in enumerate(timeline):
        if row["lang_hint"] not in {"en", "mixed"} or not row["fw_patch"]:
            continue
        next_ja = None
        for following in timeline[i + 1: i + 5]:
            if following["start_seconds"] - row["end_seconds"] > args.pair_gap_seconds:
                break
            if following["lang_hint"] == "ja" and compact(following["ja_base"]):
                next_ja = following
                break
        pairs.append({"english": row, "japanese_explanation": next_ja})

    pairs_lines = ["# English/Japanese Pair Candidates", ""]
    for pair in pairs:
        en = pair["english"]
        ja = pair["japanese_explanation"]
        pairs_lines.append(f"## chunk_{en['index']:03d} {format_time(en['start_seconds'])}-{format_time(en['end_seconds'])}")
        pairs_lines.append(f"- English: {en['fw_patch']}")
        if ja:
            pairs_lines.append(f"- Japanese explanation: {ja['ja_base']}")
        else:
            pairs_lines.append("- Japanese explanation: (not found nearby)")
        pairs_lines.append("")

    review_path = output_dir / "alternating_review_transcript.txt"
    best_path = output_dir / "alternating_best_effort.txt"
    pairs_path = output_dir / "english_japanese_pairs.md"
    candidates_path = output_dir / "alternating_candidates.json"
    summary_path = output_dir / "alternating_summary.json"
    review_path.write_text("\n".join(review_lines).strip() + "\n", encoding="utf-8")
    best_path.write_text("\n".join(best_effort_lines).strip() + "\n", encoding="utf-8")
    pairs_path.write_text("\n".join(pairs_lines).strip() + "\n", encoding="utf-8")
    candidates_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lang_counts = {}
    for row in timeline:
        lang_counts[row["lang_hint"]] = lang_counts.get(row["lang_hint"], 0) + 1
    summary = {
        "mode": "ja_en_alternating",
        "reazon_run_dir": str(reazon_run),
        "chunks": len(chunks),
        "candidates": len(candidates),
        "selected_patch_chunks": len(selected),
        "selected_indexes": sorted(selected_indexes),
        "language": args.language,
        "faster_whisper_model": args.model,
        "compute_type": args.compute_type,
        "model_load_seconds": round(model_load_seconds, 3),
        "patch_asr_seconds": round(patch_asr_seconds, 3),
        "lang_counts": lang_counts,
        "pair_candidates": len(pairs),
        "review_transcript": str(review_path),
        "best_effort": str(best_path),
        "pairs": str(pairs_path),
        "patch_results": list(patch_results.values()),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    append_log(log_path, f"lang_counts={json.dumps(lang_counts, ensure_ascii=False)}")
    append_log(log_path, "ja_en_alternating_patch_done")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
