"""Template-driven Stage 2 summarization from clean transcripts."""

import argparse
import json
import re
import sys
import time
from pathlib import Path


NO_THINK = "思考過程、推論メモ、<think>タグは出力しないでください。最終回答だけを出力してください。"

SYSTEM_PROMPT = f"""\
あなたはクリーン文字起こしを用途別の記録に変換する補助者です。{NO_THINK}
入力にある内容だけを使い、推測で補完しないでください。
"""

TEMPLATES = {
    "meeting_record": {
        "title": "面談記録",
        "filename": "meeting_record.md",
        "sections": [
            "面談の目的",
            "話し合った内容",
            "確認できたこと",
            "本人の困り感",
            "現在の支援",
            "決まったこと",
            "次にやること",
            "確認が必要な点",
        ],
    },
    "support_record": {
        "title": "学習困難支援記録",
        "filename": "support_record.md",
        "sections": [
            "困り感",
            "読み書き・聞く・話す・計算の困難",
            "有効だった支援",
            "使えそうなICT",
            "本人の希望",
            "次に試すこと",
            "確認が必要な点",
        ],
    },
    "lesson_record": {
        "title": "授業記録",
        "filename": "lesson_record.md",
        "sections": [
            "今日のテーマ",
            "大事なポイント",
            "出てきた用語",
            "先生が強調したこと",
            "宿題・提出物",
            "あとで確認すること",
        ],
    },
    "self_reflection": {
        "title": "本人の振り返り",
        "filename": "self_reflection.md",
        "sections": [
            "できたこと",
            "難しかったこと",
            "気づいたこと",
            "次に試したいこと",
            "支援してほしいこと",
            "確認が必要な点",
        ],
    },
    "meeting_memo": {
        "title": "会議メモ",
        "filename": "meeting_memo.md",
        "sections": [
            "議題",
            "決定事項",
            "未決事項",
            "担当者",
            "期限",
            "次回までのアクション",
            "確認が必要な点",
        ],
    },
    "interview_record": {
        "title": "インタビュー記録",
        "filename": "interview_record.md",
        "sections": [
            "インタビューの目的",
            "主な質問",
            "回答の要点",
            "印象的な発言",
            "固有名詞・用語",
            "確認が必要な点",
        ],
    },
}

_NO_THINK_LOGIT_BIAS = {
    151667: -100.0,
    151668: -100.0,
    248068: -100.0,
    248069: -100.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe_transcript", default="", help="Stage 1 safe_transcript.md")
    parser.add_argument("--clean_transcript", default="", help="legacy alias for --safe_transcript")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--template", default="meeting_record")
    parser.add_argument(
        "--template_file",
        default="",
        help="Optional JSON definition for one custom template",
    )
    parser.add_argument("--n_ctx", type=int, default=8192)
    parser.add_argument("--n_threads", type=int, default=4, help="CPU threads for llama_cpp inference. 4 is safe for 8GB GIGA tablets.")
    parser.add_argument("--n_batch", type=int, default=256, help="Prompt evaluation batch size. Lower values reduce peak RAM.")
    parser.add_argument("--max_chars_per_batch", type=int, default=7000)
    parser.add_argument("--max_tokens_part", type=int, default=1000)
    parser.add_argument("--max_tokens_final", type=int, default=1800)
    return parser.parse_args()


def append_log(log_path: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{stamp} {message}\n")


def print_progress(stage: str, current: int | None = None, total: int | None = None) -> None:
    """Emit one machine-readable progress line for the parent process.

    The OtoWeave app reads stdout line by line and forwards these JSON
    lines to the UI progress display."""
    payload: dict = {"stage": stage}
    if current is not None:
        payload["current"] = current
    if total is not None:
        payload["total"] = total
    print(json.dumps({"progress": payload}, ensure_ascii=False), flush=True)


# Scripts the pipeline never legitimately emits (Cyrillic, Hangul, Thai,
# Arabic, Hebrew). Qwen occasionally leaks such tokens mid-word
# (e.g. 「ファкультイ」). Greek is allowed: α/β can appear in school content.
_UNEXPECTED_SCRIPT_RE = re.compile(
    "["
    "\\u0400-\\u052f"  # Cyrillic + Cyrillic Supplement
    "\\uac00-\\ud7af"  # Hangul Syllables
    "\\u0e00-\\u0e7f"  # Thai
    "\\u0600-\\u06ff"  # Arabic
    "\\u0590-\\u05ff"  # Hebrew
    "]+"
)


def replace_unexpected_scripts(text: str) -> str:
    """Replace runs of unexpected-script characters with 「[不明]」.

    The marker is deliberate: build_review_flags collects lines containing
    [不明 into the 要確認リスト, so a garbled word surfaces for human
    review instead of being silently dropped or shipped as-is."""
    return _UNEXPECTED_SCRIPT_RE.sub("[不明]", text)


def clean_llm_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return replace_unexpected_scripts(text)


def split_text(text: str, max_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    batches: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        extra = len(paragraph) + 2
        if current and current_len + extra > max_chars:
            batches.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += extra
    if current:
        batches.append("\n\n".join(current))
    return batches or [text]


def strip_clean_transcript_notes(text: str) -> str:
    text = re.sub(r"注意:\nこのsafe transcriptは、ASR結果をチャンク順・時刻付きで確認しやすく並べた原文確認用の出力です。.*", "", text, flags=re.DOTALL)
    text = re.sub(r"注意:\nこのクリーン文字起こしはAI(?:またはプログラム)?による整形を含みます。.*", "", text, flags=re.DOTALL)
    text = re.sub(r"^#\s+クリーン文字起こし\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def compact_text(text: str) -> str:
    text = re.sub(r"^##\s+.+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[[^\]]+\]", "", text)
    return re.sub(r"\s+", "", text)


def has_substantive_content(text: str) -> bool:
    compact = compact_text(text)
    filler = {"はい", "はい。", "ありがとうございます", "はいありがとうございます", "すいません"}
    return len(compact) >= 20 and compact not in filler


def empty_template_record(template_key: str) -> str:
    template = TEMPLATES[template_key]
    lines: list[str] = []
    for section in template["sections"]:
        lines.extend([f"## {section}", "- 該当なし", ""])
    return "\n".join(lines).strip() + "\n"


# A/Bベンチ（Opus判定パネル）で、9Bモデルは忠実性で4Bに決定的に勝つ一方、
# マージ段階（part記録の統合）で自発的に過剰圧縮し情報を落とすことが確認された。
# マージ系のstageにこの省略禁止指示を追記すると網羅性が54%→94%に回復した。
# part段階の指示（テンプレートごとのinstruction）は変えず、stage判定でのみ
# マージ系プロンプトに追記する。
_MERGE_STAGE_KEYWORDS = ("merge round", "final integration")

_MERGE_NO_OMISSION_INSTRUCTION = (
    "複数のpart記録を統合する。各partに含まれる情報・箇条書きは一切省略せず、"
    "同じ内容の重複だけを1つに統合して、全ての項目をそのまま保持する。"
    "短くまとめ直すことは禁止。"
)


def template_prompt(template_key: str, transcript: str, stage: str) -> str:
    template = TEMPLATES[template_key]
    sections = "\n".join(f"## {section}" for section in template["sections"])
    dictionary = str(template.get("dictionary", "")).strip()
    dictionary_block = (
        f"\n参考辞書:\n{dictionary}\n"
        if dictionary
        else ""
    )
    instruction = template.get("instruction", "指定された見出しに沿って整理する")
    if any(keyword in stage for keyword in _MERGE_STAGE_KEYWORDS):
        instruction = f"{instruction}\n{_MERGE_NO_OMISSION_INSTRUCTION}"
    return f"""\
safe_transcript.mdを、用途別テンプレートに沿って整理してください。

用途: {template["title"]}
追加の指示: {instruction}
処理段階: {stage}
{dictionary_block}

重要ルール:
1. 入力にない内容を追加しない
2. 推測で補完しない
3. 「検討した」を「決定した」に変えない
4. 不明な語・曖昧な語は「確認が必要な点」に残す
5. 個人名・学校名・制度名を勝手に修正しない
6. 各項目は短い箇条書きにする
7. 該当がない見出しには「- 該当なし」と書く
8. 見出し名は下記の「## ...」をそのまま使い、言い換えない
9. 見出しを本文中の「- 重要なポイント:」のようなラベルにしない

出力見出し:
{sections}

safe transcript:
{transcript}

出力:
/no_think
"""


def count_tokens(llm, text: str) -> int:
    """Token count via the model tokenizer; character-count fallback."""
    try:
        return len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))
    except Exception:
        return max(1, len(text))


def split_batches_by_tokens(llm, batches: list[str], budget_tokens: int) -> list[str]:
    """Re-split character-based batches so each fits the token budget."""
    result: list[str] = []
    for batch in batches:
        if count_tokens(llm, batch) <= budget_tokens:
            result.append(batch)
            continue
        current: list[str] = []
        current_tokens = 0
        for paragraph in re.split(r"\n{2,}", batch):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            tokens = count_tokens(llm, paragraph) + 2
            if tokens > budget_tokens:
                # A single oversized paragraph: hard-split proportionally.
                size = max(200, int(len(paragraph) * budget_tokens / tokens))
                pieces = [paragraph[i:i + size] for i in range(0, len(paragraph), size)]
            else:
                pieces = [paragraph]
            for piece in pieces:
                piece_tokens = count_tokens(llm, piece) + 2
                if current and current_tokens + piece_tokens > budget_tokens:
                    result.append("\n\n".join(current))
                    current = []
                    current_tokens = 0
                current.append(piece)
                current_tokens += piece_tokens
        if current:
            result.append("\n\n".join(current))
    return result or batches


def call_llm(llm, prompt: str, max_tokens: int) -> tuple[str, str]:
    """Run one completion. Returns (clean_text, finish_reason)."""
    vocab_size = llm.n_vocab()
    logit_bias = {k: v for k, v in _NO_THINK_LOGIT_BIAS.items() if k < vocab_size}
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        repeat_penalty=1.12,
        max_tokens=max_tokens,
        logit_bias=logit_bias,
    )
    choice = response["choices"][0]
    finish_reason = str(choice.get("finish_reason", ""))
    return clean_llm_text(choice["message"]["content"]), finish_reason


_META_NOTE_LINE_RE = re.compile(
    r"^-?\s*[（(][^）)]*(?:入力内容|明示的)[^）)]*なし[）)]\s*$"
)
_GARBLED_NA_LINE_RE = re.compile(r"^-\s*-\s*該当なし\s*$")


def strip_part_meta_notes(text: str) -> str:
    """Drop part-record noise that should never reach a merge prompt.

    9B part records sometimes add meta-commentary about the input itself,
    e.g. "- （入力内容に明示的な決定事項は記載なし）", instead of just
    using the template's own「- 該当なし」convention. Feeding that
    commentary into the merge prompt nudges the model toward reporting on
    the absence of information rather than merging content, so it is
    stripped before parts are merged. Formatting drift also occasionally
    doubles the placeholder into a garbled "- - 該当なし" line; that is
    dropped too (the real placeholder is re-added by
    fill_empty_template_sections/ensure_template_sections downstream).
    """
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if _META_NOTE_LINE_RE.match(stripped) or _GARBLED_NA_LINE_RE.match(stripped):
            continue
        lines.append(line)
    return "\n".join(lines)


def merge_part_records(
    llm,
    parts: list[str],
    template_key: str,
    args,
    logs_path: Path,
) -> tuple[str, bool]:
    """Merge part records hierarchically so every merge prompt fits n_ctx.

    Returns (final_record, truncated) where truncated reports whether any
    merge output hit the token limit.
    """
    parts = [strip_part_meta_notes(item) for item in parts]
    prompt_overhead = count_tokens(
        llm, template_prompt(template_key, "", "final integration")
    ) + 64
    merge_budget = max(512, args.n_ctx - args.max_tokens_final - prompt_overhead)
    # Bound every item to half the budget so any two items always fit in one
    # merge call and each round is guaranteed to shrink the list.
    item_limit = max(256, merge_budget // 2)
    truncated = False

    def clamp(item: str) -> str:
        nonlocal truncated
        tokens = count_tokens(llm, item)
        if tokens <= item_limit:
            return item
        truncated = True
        return item[: max(200, int(len(item) * item_limit / tokens))]

    round_items = [clamp(item) for item in parts]
    round_number = 0
    while len(round_items) > 1:
        round_number += 1
        groups: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        for item in round_items:
            tokens = count_tokens(llm, item) + 2
            if current and current_tokens + tokens > merge_budget:
                groups.append(current)
                current = []
                current_tokens = 0
            current.append(item)
            current_tokens += tokens
        if current:
            groups.append(current)
        next_items: list[str] = []
        for group_index, group in enumerate(groups, start=1):
            if len(group) == 1:
                next_items.append(group[0])
                continue
            joined = "\n\n".join(group)
            merged, finish_reason = call_llm(
                llm,
                template_prompt(
                    template_key,
                    joined,
                    "final integration"
                    if len(groups) == 1
                    else f"merge round {round_number}",
                ),
                args.max_tokens_final,
            )
            if finish_reason == "length":
                truncated = True
            merged = clamp(ensure_template_sections(merged, template_key))
            append_log(
                logs_path,
                f"merge_round_{round_number}_group_{group_index:02d}"
                f" inputs={len(group)} finish_reason={finish_reason}",
            )
            next_items.append(merged)
        round_items = next_items
    return round_items[0], truncated


def ensure_template_sections(text: str, template_key: str) -> str:
    template = TEMPLATES[template_key]
    result = normalize_template_output(re.sub(r"^<!--.*?-->\s*", "", text.strip(), flags=re.MULTILINE), template_key)
    result = sanitize_template_record(result)
    for section in template["sections"]:
        if not re.search(rf"^##\s+{re.escape(section)}\s*$", result, flags=re.MULTILINE):
            result += f"\n\n## {section}\n- 該当なし"
    result = fill_empty_template_sections(result, template_key)
    return result.strip() + "\n"


def fill_empty_template_sections(text: str, template_key: str) -> str:
    template = TEMPLATES[template_key]
    lines = text.splitlines()
    result: list[str] = []
    current_section = ""
    current_content: list[str] = []

    def flush() -> None:
        if not current_section:
            result.extend(current_content)
            return
        result.append(f"## {current_section}")
        body = [line for line in current_content if line.strip()]
        if not body:
            result.append("- 該当なし")
        else:
            result.extend(current_content)
        result.append("")

    for line in lines:
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading and heading.group(1) in template["sections"]:
            flush()
            current_section = heading.group(1)
            current_content = []
        else:
            current_content.append(line)
    flush()
    return "\n".join(result)


def sanitize_template_record(text: str) -> str:
    """Remove tool/instruction artifacts and soften speculative wording."""
    forbidden_fragments = (
        "safe_transcript",
        "clean_transcript",
        "用途別テンプレート",
        "テンプレートに沿って",
        "メモを作成",
        "上記の内容を整理",
        "入力内容",
    )
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and any(fragment in stripped for fragment in forbidden_fragments):
            continue
        if _GARBLED_NA_LINE_RE.match(stripped):
            # The model occasionally emits this doubled placeholder itself
            # during merging (not just inherited from part inputs handled
            # by strip_part_meta_notes); fill_empty_template_sections
            # re-adds a clean "- 該当なし" if the section ends up empty.
            continue
        if "一般的な知識" in stripped:
            line = "- 資格要件の詳細は要確認。"
        line = line.replace("から推測される", "として記録されている")
        line = line.replace("推測される", "記録されている")
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def normalize_template_output(text: str, template_key: str) -> str:
    """Repair common Qwen formatting drift in template records."""
    template = TEMPLATES[template_key]
    aliases = {
        "重要なポイント": "大事なポイント",
        "確認が必要な点": "あとで確認すること" if template_key == "lesson_record" else "確認が必要な点",
    }
    section_names = set(template["sections"])
    section_names.update(aliases)
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        label_match = re.match(r"^-?\s*([^:：]+)\s*[:：]\s*$", stripped)
        if label_match:
            label = label_match.group(1).strip()
            target = aliases.get(label, label)
            if target in template["sections"]:
                lines.append(f"## {target}")
                continue
        for section in section_names:
            target = aliases.get(section, section)
            if target in template["sections"] and re.match(rf"^-?\s*{re.escape(section)}\s*[:：]\s*$", stripped):
                lines.append(f"## {target}")
                break
        else:
            # Normalize nested bullets generated under label-style sections.
            line = re.sub(r"^\s+-\s+", "- ", line)
            lines.append(line.rstrip())
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def build_review_flags(record: str) -> str:
    flags: list[str] = []
    in_check = False
    for line in record.splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            title = heading.group(1)
            in_check = title in {"確認が必要な点", "あとで確認すること"}
            continue
        stripped = line.strip()
        if in_check and stripped.startswith("- ") and "該当なし" not in stripped:
            flags.append(stripped)
        elif "[要確認" in stripped or "[不明" in stripped:
            flags.append(f"- {stripped.lstrip('- ').strip()}")
    if not flags:
        flags = ["- 目立つ要確認箇所はありません。"]
    return "# 要確認リスト\n\n" + "\n".join(dict.fromkeys(flags)) + "\n"


def add_record_note(text: str, truncated: bool = False) -> str:
    lines = [
        "注意:",
        "この記録は、safe transcriptとAI整形をもとにした確認用メモです。",
        "正式記録として使用する場合は、原文文字起こしおよび元音声と照合してください。",
    ]
    if truncated:
        # Hidden truncation would read as "the class had nothing more" —
        # tell the reader explicitly that content may be missing.
        lines.append(
            "⚠ AIの出力が長さの上限に達したため、内容の一部が欠けている可能性があります。"
        )
    note = "\n".join(lines)
    if "正式記録として使用する場合は、原文文字起こし" in text:
        if truncated and "内容の一部が欠けている可能性" not in text:
            return text.rstrip() + "\n" + lines[-1] + "\n"
        return text.rstrip() + "\n"
    return text.rstrip() + "\n\n" + note + "\n"


def record_sections_all_empty(record: str, template_key: str) -> bool:
    """True when every template section contains only 「該当なし」."""
    template = TEMPLATES[template_key]
    sections = set(template["sections"])
    current = ""
    bodies: dict[str, list[str]] = {section: [] for section in sections}
    for line in record.splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            current = heading.group(1) if heading.group(1) in sections else ""
            continue
        stripped = line.strip()
        if current and stripped and not stripped.startswith("注意:"):
            bodies[current].append(stripped)
    def only_empty(items: list[str]) -> bool:
        return all("該当なし" in item for item in items) if items else True
    return all(only_empty(items) for items in bodies.values())


def main() -> int:
    args = parse_args()
    if args.template_file:
        value = json.loads(
            Path(args.template_file).read_text(encoding="utf-8")
        )
        TEMPLATES[args.template] = {
            "title": str(value["name"]),
            "filename": f"{args.template}.md",
            "sections": list(value.get("sections") or ["要約"]),
            "instruction": str(value.get("instruction", "")),
            "dictionary": str(value.get("dictionary", "")),
        }
    if args.template not in TEMPLATES:
        raise SystemExit(f"Unknown template: {args.template}")
    transcript_arg = args.safe_transcript or args.clean_transcript
    if not transcript_arg:
        raise SystemExit("--safe_transcript is required")
    clean_path = Path(transcript_arg)
    output_dir = Path(args.output_dir)
    summaries_dir = output_dir / "summaries"
    logs_path = Path(args.log)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    clean_text = strip_clean_transcript_notes(clean_path.read_text(encoding="utf-8", errors="replace"))
    batches = split_text(clean_text, args.max_chars_per_batch)

    append_log(logs_path, "template_summarize_start")
    append_log(logs_path, f"template={args.template}")
    append_log(logs_path, f"safe_transcript={clean_path}")
    append_log(logs_path, "thinking_mode=off")
    append_log(logs_path, f"batches={len(batches)}")

    if not has_substantive_content(clean_text):
        append_log(logs_path, "no_substantive_content=true")
        final_record = add_record_note(empty_template_record(args.template))
        template = TEMPLATES[args.template]
        record_path = summaries_dir / template["filename"]
        record_path.write_text(final_record, encoding="utf-8")
        review_flags_path = summaries_dir / f"{args.template}_review_flags.md"
        review_flags_path.write_text(build_review_flags(final_record), encoding="utf-8")
        metadata = {
            "template": args.template,
            "template_title": template["title"],
            "safe_transcript": str(clean_path),
            "clean_transcript": str(clean_path),
            "record": str(record_path),
            "review_flags": str(review_flags_path),
            "thinking_mode": "off",
            "batches": 0,
            "model": args.model,
            "model_load_seconds": 0,
            "final_seconds": 0,
            "local_only": True,
            "needs_human_review": True,
        }
        metadata_path = summaries_dir / f"{args.template}_metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        append_log(logs_path, "template_summarize_done_no_content")
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
        return 0

    from llama_cpp import Llama

    print(f"Loading model: {args.model}", flush=True)
    print_progress("load")
    load_start = time.perf_counter()
    llm = Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_batch=args.n_batch,
        verbose=False,
    )
    load_seconds = time.perf_counter() - load_start
    append_log(logs_path, f"model_load_seconds={load_seconds:.3f}")

    # Re-split with the real tokenizer so every part prompt fits n_ctx.
    part_prompt_overhead = count_tokens(
        llm, template_prompt(args.template, "", "part 1")
    ) + 64
    part_budget = max(512, args.n_ctx - args.max_tokens_part - part_prompt_overhead)
    batches = split_batches_by_tokens(llm, batches, part_budget)
    append_log(logs_path, f"token_batches={len(batches)} part_budget_tokens={part_budget}")

    truncated = False
    part_dir = summaries_dir / f"{args.template}_parts"
    part_dir.mkdir(exist_ok=True)
    part_outputs: list[str] = []
    for idx, batch in enumerate(batches, start=1):
        print(f"Summarizing template part {idx}/{len(batches)}", flush=True)
        print_progress("part", current=idx, total=len(batches))
        start = time.perf_counter()
        part, finish_reason = call_llm(llm, template_prompt(args.template, batch, f"part {idx}"), args.max_tokens_part)
        if finish_reason == "length":
            truncated = True
        part = ensure_template_sections(part, args.template)
        seconds = time.perf_counter() - start
        append_log(
            logs_path,
            f"part_{idx:03d}_seconds={seconds:.3f} chars_in={len(batch)}"
            f" chars_out={len(part)} finish_reason={finish_reason}",
        )
        (part_dir / f"part_{idx:03d}.md").write_text(part, encoding="utf-8")
        part_outputs.append(part)

    final_start = time.perf_counter()
    if len(part_outputs) == 1:
        final_record = part_outputs[0]
    else:
        print_progress("merge")
        final_record, merge_truncated = merge_part_records(
            llm,
            part_outputs,
            args.template,
            args,
            logs_path,
        )
        truncated = truncated or merge_truncated
    final_record = ensure_template_sections(final_record, args.template)
    if record_sections_all_empty(final_record, args.template):
        # The transcript had substance (checked above) yet every section is
        # 「該当なし」: the model output was unusable (e.g. all thinking
        # text). Fail instead of caching a fake "nothing important" record.
        append_log(logs_path, "all_sections_empty=true -> fail")
        print(
            "要約の生成に失敗しました（AIの出力から内容を取り出せませんでした）。"
            "もう一度お試しください。",
            file=sys.stderr,
        )
        return 3
    final_record = add_record_note(final_record, truncated=truncated)
    final_seconds = time.perf_counter() - final_start
    append_log(logs_path, f"final_seconds={final_seconds:.3f} chars_out={len(final_record)} truncated={truncated}")

    template = TEMPLATES[args.template]
    record_path = summaries_dir / template["filename"]
    record_path.write_text(final_record, encoding="utf-8")
    review_flags_path = summaries_dir / f"{args.template}_review_flags.md"
    review_flags_path.write_text(build_review_flags(final_record), encoding="utf-8")

    metadata = {
        "template": args.template,
        "template_title": template["title"],
        "safe_transcript": str(clean_path),
        "clean_transcript": str(clean_path),
        "record": str(record_path),
        "review_flags": str(review_flags_path),
        "thinking_mode": "off",
        "batches": len(batches),
        "model": args.model,
        "model_load_seconds": round(load_seconds, 3),
        "final_seconds": round(final_seconds, 3),
        "n_ctx": args.n_ctx,
        "truncated": truncated,
        "local_only": True,
        "needs_human_review": True,
    }
    metadata_path = summaries_dir / f"{args.template}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    append_log(logs_path, "template_summarize_done")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
