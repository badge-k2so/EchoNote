"""Compare a local ASR transcript with a reference transcript."""

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


KEYWORDS = [
    "ディズニー",
    "Disneyland",
    "Disney",
    "血圧",
    "体温",
    "寝てる",
    "早朝",
    "マレーシア",
    "怪我",
    "けが",
    "顎",
    "あご",
    "chin",
    "ground",
    "jump",
    "shoulder",
    "catch",
    "medicine",
    "pain",
    "doctor",
    "32kg",
    "32 kg",
    "kg",
    "bear",
    "Kyoto",
    "hotel",
    "sports director",
    "water",
    "syrup",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate_review", default="")
    parser.add_argument("--hybrid_summary", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def strip_markup(text: str) -> str:
    text = re.sub(r"\[SPEAKER_\d+\]", "", text)
    text = re.sub(r"^={3,}.*={3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\[(ja_base|fw_patch)\]\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[\d{1,3}:\d{2}\.\d{2}\s+-->\s+\d{1,3}:\d{2}\.\d{2}\]", "", text)
    text = re.sub(r"\[\d+\.\d+\s+-->\s+\d+\.\d+\]", "", text)
    return text


def normalize_for_similarity(text: str) -> str:
    text = strip_markup(text).lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[、。，．,.!?！？:：;；「」『』（）()\[\]【】\-_=]+", "", text)
    return text


def count_latin_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z][A-Za-z']+", text))


def count_japanese_chars(text: str) -> int:
    return len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", text))


def keyword_hits(text: str) -> dict[str, bool]:
    lower = text.lower()
    return {kw: (kw.lower() in lower) for kw in KEYWORDS}


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> int:
    args = parse_args()
    reference = read_text(args.reference)
    candidate = read_text(args.candidate)
    review = read_text(args.candidate_review) if args.candidate_review else ""
    hybrid_summary = {}
    if args.hybrid_summary:
        hybrid_summary = json.loads(read_text(args.hybrid_summary))

    ref_norm = normalize_for_similarity(reference)
    cand_norm = normalize_for_similarity(candidate)
    review_norm = normalize_for_similarity(review) if review else ""

    ratio = SequenceMatcher(None, ref_norm, cand_norm).ratio() if ref_norm and cand_norm else 0.0
    review_ratio = SequenceMatcher(None, ref_norm, review_norm).ratio() if ref_norm and review_norm else None

    ref_hits = keyword_hits(reference)
    cand_hits = keyword_hits(candidate)
    review_hits = keyword_hits(review) if review else {}

    hit_rows = []
    for kw in KEYWORDS:
        hit_rows.append(
            {
                "keyword": kw,
                "reference": ref_hits[kw],
                "candidate": cand_hits[kw],
                "review": review_hits.get(kw, False),
            }
        )

    ref_hit_count = sum(ref_hits.values())
    cand_hit_count = sum(cand_hits[k] for k in KEYWORDS if ref_hits[k])
    review_hit_count = sum(review_hits.get(k, False) for k in KEYWORDS if ref_hits[k]) if review else 0

    metrics = {
        "reference_chars": len(strip_markup(reference)),
        "candidate_chars": len(strip_markup(candidate)),
        "review_chars": len(strip_markup(review)) if review else 0,
        "reference_japanese_chars": count_japanese_chars(reference),
        "candidate_japanese_chars": count_japanese_chars(candidate),
        "reference_latin_words": count_latin_words(reference),
        "candidate_latin_words": count_latin_words(candidate),
        "review_latin_words": count_latin_words(review) if review else 0,
        "similarity_best_effort": round(ratio, 4),
        "similarity_review": round(review_ratio, 4) if review_ratio is not None else None,
        "reference_keyword_count": ref_hit_count,
        "candidate_keyword_hits": cand_hit_count,
        "review_keyword_hits": review_hit_count if review else None,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Hybrid ASR vs WhisperX large-v3 comparison")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Reference: `{args.reference}`")
    lines.append(f"- Hybrid best effort: `{args.candidate}`")
    if args.candidate_review:
        lines.append(f"- Hybrid review: `{args.candidate_review}`")
    if args.hybrid_summary:
        lines.append(f"- Hybrid summary: `{args.hybrid_summary}`")
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    if hybrid_summary:
        lines.append(f"- Chunks: {hybrid_summary.get('chunks')}")
        lines.append(f"- Candidates: {hybrid_summary.get('candidates')}")
        lines.append(f"- Patched chunks: {hybrid_summary.get('selected_patch_chunks')}")
        lines.append(f"- faster-whisper patch ASR seconds: {hybrid_summary.get('patch_asr_seconds')}")
        lines.append(f"- Hybrid patch script seconds: {hybrid_summary.get('total_seconds')}")
    lines.append("")
    lines.append("## Text Metrics")
    lines.append("")
    lines.append(f"- Reference chars: {metrics['reference_chars']}")
    lines.append(f"- Hybrid best-effort chars: {metrics['candidate_chars']}")
    lines.append(f"- Hybrid review chars: {metrics['review_chars']}")
    lines.append(f"- Reference Latin words: {metrics['reference_latin_words']}")
    lines.append(f"- Hybrid best-effort Latin words: {metrics['candidate_latin_words']}")
    lines.append(f"- Hybrid review Latin words: {metrics['review_latin_words']}")
    lines.append(f"- Similarity, best-effort vs reference: {pct(metrics['similarity_best_effort'])}")
    if metrics["similarity_review"] is not None:
        lines.append(f"- Similarity, review vs reference: {pct(metrics['similarity_review'])}")
    lines.append(f"- Keyword hits in reference set: {metrics['reference_keyword_count']}")
    lines.append(f"- Keyword hits by best-effort: {metrics['candidate_keyword_hits']}")
    if metrics["review_keyword_hits"] is not None:
        lines.append(f"- Keyword hits by review: {metrics['review_keyword_hits']}")
    lines.append("")
    lines.append("## Keyword Coverage")
    lines.append("")
    lines.append("| Keyword | Reference | Best effort | Review |")
    lines.append("|---|---:|---:|---:|")
    for row in hit_rows:
        if not row["reference"] and not row["candidate"] and not row["review"]:
            continue
        lines.append(
            f"| {row['keyword']} | {'yes' if row['reference'] else 'no'} | "
            f"{'yes' if row['candidate'] else 'no'} | {'yes' if row['review'] else 'no'} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `hybrid_best_effort.txt` only replaces a Japanese-base chunk when the faster-whisper patch contains Latin words.")
    lines.append("- `hybrid_review_transcript.txt` keeps both `ja_base` and `fw_patch`, so it is better for human review and debugging.")
    lines.append("- Speaker diarization is not produced by this CPU hybrid pipeline.")
    lines.append("- Sequence similarity is a rough character-level signal, not an ASR accuracy score.")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = output.with_suffix(".json")
    json_path.write_text(
        json.dumps({"metrics": metrics, "keywords": hit_rows, "hybrid_summary": hybrid_summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
