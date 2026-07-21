from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path


CHUNK_HEADER_RE = re.compile(r"^=+\s*chunk[^=]*=+$", re.IGNORECASE)
SYSTEM_LINE_RE = re.compile(
    r"^\s*(?:\[(?:ja_base|fw_patch|parakeet_patch|patch_rejected)\]|"
    r"\[?\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?\s*--?>.*?\]?)\s*$",
    re.IGNORECASE,
)
DATE_PATTERNS = (
    re.compile(r"(?<!\d)(20\d{2})[-_.年](\d{1,2})[-_.月](\d{1,2})(?:日)?(?!\d)"),
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"),
)
FILLER_PREFIX_RE = re.compile(
    r"^(?:(?:えー+|えっと|あの+|その+|まあ+|はい|では|じゃあ|さて|今日は|今回|"
    r"よろしくお願いします|おはようございます|こんにちは)[、,。\s]*)+"
)
NOISE_ONLY_RE = re.compile(
    r"^(?:録音(?:中|を開始しました)?|recording in progress|ありがとうございます|"
    r"よろしくお願いします|はい|うん|あっ|あれ|えー+|無音|empty)[。.!！\s]*$",
    re.IGNORECASE,
)
CONTENT_TOKEN_RE = re.compile(
    r"[一-龯々〆ヵヶ]{2,12}|[ァ-ヴー]{3,20}|[A-Za-z][A-Za-z'-]{3,20}"
)
WINDOWS_INVALID_RE = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")
GENERIC_AUDIO_LABEL_RE = re.compile(
    r"^(?:audio|recording|record|voice|memo|録音|音声|ボイスメモ|新規録音)\d*$",
    re.IGNORECASE,
)
TOPIC_TERM_RE = re.compile(
    r"(?:[一-龯々]{1,10}|[ァ-ヴー]{2,18}|[A-Za-z][A-Za-z'-]{1,18})"
    r"(?:活動|受験|面談|会議|講義|授業|支援|教育|学習|研究|実験|発表|相談|研修|"
    r"検査|評価|課題|要因|配慮|障害|困難|プロジェクト|プログラム|インタビュー|テスト)"
)
TOPIC_INTRO_RE = re.compile(
    r"(?:について(?:話|説明|学習|勉強|検討)|(?:今日|今回|本日)の(?:内容|テーマ)|"
    r"(?:授業|講義|面談|会議|勉強会)で(?:は|扱)|(?:テーマ|議題|内容)(?:は|として))",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suggest a recording filename from its date and transcript")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recorded_at", default="", help="Optional ISO date/time override")
    parser.add_argument("--max_title_chars", type=int, default=36)
    return parser.parse_args()


def extract_recorded_date(audio_path: Path, recorded_at: str = "") -> tuple[dt.date, str]:
    if recorded_at:
        value = dt.datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
        return value.date(), "provided_recorded_at"

    for pattern in DATE_PATTERNS:
        match = pattern.search(audio_path.stem)
        if not match:
            continue
        try:
            return dt.date(*(int(part) for part in match.groups())), "audio_filename"
        except ValueError:
            pass

    timestamp = audio_path.stat().st_ctime
    return dt.datetime.fromtimestamp(timestamp).date(), "file_creation_time"


def transcript_segments(text: str) -> list[str]:
    cleaned_lines: list[str] = []
    for raw_line in text.replace("\r", "").split("\n"):
        line = raw_line.strip()
        if not line or CHUNK_HEADER_RE.match(line) or SYSTEM_LINE_RE.match(line):
            continue
        line = re.sub(r"^\[[^\]]+\]\s*", "", line)
        if line and line.lower() != "(empty)":
            cleaned_lines.append(line)

    joined = "。".join(cleaned_lines)
    parts = re.split(r"(?<=[。！？!?])|[\n]+", joined)
    segments: list[str] = []
    for part in parts:
        segment = re.sub(r"\s+", " ", part).strip(" 。、,\t")
        segment = FILLER_PREFIX_RE.sub("", segment).strip(" 。、,\t")
        if len(segment) < 6 or NOISE_ONLY_RE.match(segment):
            continue
        segments.append(segment)
    return segments


def segment_score(segment: str, index: int) -> float:
    tokens = CONTENT_TOKEN_RE.findall(segment)
    unique_tokens = {token.casefold() for token in tokens}
    score = len(unique_tokens) * 8.0 + min(len(segment), 36) * 0.45
    if 10 <= len(segment) <= 48:
        score += 5.0
    if segment.endswith(("?", "？")):
        score -= 5.0
    if re.match(r"^(?:私|僕|俺|we|i)\b", segment, flags=re.IGNORECASE):
        score -= 2.0
    # Topic-introduction sentences are usually better filenames than a later
    # list of details. This is structural wording, not a subject vocabulary.
    if TOPIC_INTRO_RE.search(segment):
        score += 18.0
    score -= min(index, 20) * 1.5
    return score


def extract_filename_label(audio_path: Path) -> str:
    label = audio_path.stem
    for pattern in DATE_PATTERNS:
        label = pattern.sub(" ", label)
    label = re.sub(r"(?<!\d)\d{1,2}[_:.時-]\d{1,2}(?:[_:.分-]\d{1,2})?(?:秒)?", " ", label)
    label = re.sub(r"^[\s_.-]+|[\s_.-]+$", "", label)
    label = re.sub(r"[\s_.-]+", " ", label)
    if not label or GENERIC_AUDIO_LABEL_RE.fullmatch(label.replace(" ", "")):
        return ""
    if not re.search(r"[一-龯ぁ-んァ-ヴA-Za-z]", label):
        return ""
    return sanitize_title(label)


def extract_topic_terms(segments: list[str], max_terms: int = 2) -> list[str]:
    stats: dict[str, tuple[int, int]] = {}
    for segment_index, segment in enumerate(segments[:40]):
        for match in TOPIC_TERM_RE.finditer(segment):
            term = re.sub(
                r"^(?:多分|たぶん|例えば|たとえば|主な|今回の|今日の|本日の)+",
                "",
                match.group(0).strip(),
            )
            if len(term) < 3:
                continue
            count, first_index = stats.get(term, (0, segment_index))
            stats[term] = (count + 1, min(first_index, segment_index))

    ranked = sorted(
        stats,
        key=lambda term: (-stats[term][0], stats[term][1], -len(term), term),
    )
    selected: list[str] = []
    for term in ranked:
        if any(term in chosen or chosen in term for chosen in selected):
            continue
        selected.append(term)
        if len(selected) >= max_terms:
            break
    return selected


def sanitize_title(value: str, max_chars: int = 36) -> str:
    title = FILLER_PREFIX_RE.sub("", value)
    title = re.sub(r"\s+", " ", title)
    title = WINDOWS_INVALID_RE.sub("", title)
    title = title.strip(" .。_-")
    if len(title) > max_chars:
        shortened = title[:max_chars]
        boundary = max(shortened.rfind("、"), shortened.rfind(","), shortened.rfind(" "))
        if boundary >= max_chars // 2:
            shortened = shortened[:boundary]
        title = shortened.rstrip(" 、,")
    return title or "録音"


def extract_title(transcript_text: str, max_chars: int = 36) -> tuple[str, str]:
    segments = transcript_segments(transcript_text)
    if not segments:
        return "録音", "fallback_no_meaningful_text"
    topic_terms = extract_topic_terms(segments)
    if len(topic_terms) >= 2:
        return sanitize_title("・".join(topic_terms), max_chars=max_chars), "transcript_topic_terms"
    candidates = segments[:24]
    best_index, best = max(
        enumerate(candidates), key=lambda item: segment_score(item[1], item[0])
    )
    return sanitize_title(best, max_chars=max_chars), f"transcript_segment_{best_index + 1}"


def build_filename_suggestion(
    audio_path: Path,
    transcript_text: str,
    recorded_at: str = "",
    max_title_chars: int = 36,
) -> dict:
    recorded_date, date_source = extract_recorded_date(audio_path, recorded_at)
    filename_label = extract_filename_label(audio_path)
    if filename_label:
        title = sanitize_title(filename_label, max_chars=max_title_chars)
        title_source = "audio_filename_label"
    else:
        title, title_source = extract_title(transcript_text, max_chars=max_title_chars)
    stem = sanitize_title(f"{recorded_date.isoformat()}_{title}", max_chars=80)
    return {
        "recorded_date": recorded_date.isoformat(),
        "date_source": date_source,
        "title": title,
        "title_source": title_source,
        "original_audio_filename": audio_path.name,
        "suggested_audio_filename": f"{stem}{audio_path.suffix.lower()}",
        "suggested_transcript_filename": f"{stem}.txt",
        "requires_user_confirmation": True,
    }


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio)
    transcript_path = Path(args.transcript)
    output_path = Path(args.output)
    suggestion = build_filename_suggestion(
        audio_path,
        transcript_path.read_text(encoding="utf-8", errors="replace"),
        recorded_at=args.recorded_at,
        max_title_chars=args.max_title_chars,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(suggestion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(suggestion, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
