"""School-oriented LLM post-processing for hybrid ASR transcripts."""

import argparse
import json
import re
import time
from pathlib import Path


NO_THINK = "思考過程、推論メモ、<think>タグは出力しないでください。最終回答だけを出力してください。"

SYSTEM_PROMPT = f"""\
あなたは学習支援記録を短く正確に整理するプロの要約者です。{NO_THINK}
入力にある内容だけを使い、入力にない場面名・役割名・説明を追加しないでください。
重要語、困りごと、支援方法、確認事項は落とさず、冗長な話し言葉だけを圧縮してください。
"""

PART_USER_PROMPT = """\
音声記録の整形メモを作成します。

入力形式:
- 「# クリーン文字起こし」で始まる音声記録
- 各発話は「## chunk_NNN 時刻」で区切られている
- 「認識: fw_patch」または「認識: parakeet_patch」のある発話 = 英語補助ASRで認識された発話

重要ルール:
1. ## 整形記録 にタイムスタンプ・chunk番号・「認識: fw_patch」「認識: parakeet_patch」などのシステム用語を書かない
2. 英語補助ASRのある発話は内容が英語なら ## 英語・専門語・固有名詞 か ## 要確認箇所 に入れる
3. 意味のない相槌のみの発話は省略
4. 入力にない場面名・役割名を追加しない
5. 形式的な説明・日程・手順だけを優先しない。本人の理由、体験、感想、意見、希望、学びたいこと、伝えたいことが含まれる発話は必ず ## 整形記録 に要約文として書く
6. ## 整形記録 の各項目は必ず30〜90字の要約文にする。発話の原文をそのままコピーしない。内容が不明瞭な発話は ## 要確認箇所 に入れる

出力形式（必ず以下の3つの見出しを使い、発話内容の要約文を書く）:

## 整形記録
（意味ある発話・質問・決定の要約文。タイムスタンプ・chunk番号なし。最大10項目。ルール: ①1項目30〜90字の要約文で書く ②具体例・固有名詞・数字を含める ③生の発話をそのまま貼らない ④1項目に複数話題を詰め込まない ⑤前半・中盤・後半から偏らず拾う）

## 英語・専門語・固有名詞
（英語の句・文・固有名詞。単語のみは不可。最大10項目）

## 要確認箇所
（[タイムスタンプ] 発話。判断できない箇所のみ。最大8項目）

入力:
{transcript}

出力:
/no_think
"""

FINAL_USER_PROMPT = """\
複数のpart整形結果を1つの記録に統合してください。

【ルール】
1. 重複・相づち・意味の薄い断片は省く
2. 各partの内容を全部並べ直さない（重要な発話・決定・質問を選んで1項目1文でまとめる）
3. 入力にない場面名・役割名を作らない
4. 英語発話・固有名詞・数字は保持する
5. 不確かな箇所は省かず要確認として残す
6. ## 整形記録 にタイムスタンプを書かない
7. 1項目は30〜100字の要約文にする。話し言葉のままにしない

【出力形式（必ずこの3つの見出しを使う）】

## 整形記録
（最大10項目。1項目=1文、タイムスタンプなし）

## 英語・専門語・固有名詞
（英語表現・専門語・固有名詞を最大15項目）

## 要確認箇所
（[タイムスタンプ] 発話を最大10項目）

【part整形結果】
{partial_records}

【統合結果】
/no_think
"""

LEARNER_SUMMARY_PROMPT = """\
以下は学校場面の授業・面談・勉強会・会議などの文字起こし・小要約です。
誤字や聞き間違いが含まれる可能性があります。

目的:
小学生・中学生があとで復習しやすいように、短く整理してください。

守ること:
- 事実を追加しない
- わからない内容は推測しない
- 人名や学校名は原文どおりに残す
- AIの判断で内容を変えない
- 短く書く
- 箇条書きは各見出し最大5項目
- 本題語、困りごと、支援方法、確認事項は落とさない
- 要確認語や意味不明な断片を、今日の内容・大事なことに入れない

出力:
## 今日の内容
## 大事なこと
## わからなかったかもしれないこと
## 次に先生や大人に確認すること

文字起こし:
{transcript}

出力:
/no_think
"""

LEARNER_FINAL_PROMPT = """\
以下は学校場面の授業・面談・勉強会・会議などの小要約をさらに短くしたものです。
誤字や聞き間違いが含まれる可能性があります。

目的:
小学生・中学生があとで復習しやすいように、全体を短く整理してください。

守ること:
- 事実を追加しない
- わからない内容は推測しない
- 人名や学校名は原文どおりに残す
- AIの判断で内容を変えない
- 短く、ただし講座全体の流れがわかるように書く
- 同じ内容はまとめる
- 冒頭だけに偏らず、後半に出てきた重要内容も残す
- 要確認語や意味不明な断片を、今日の内容・大事なことに入れない
- 原文にない説明を追加しない
- 「AIが判断した」「AIが指導した」「AIが解決した」のように書かない

内容の優先順位:
1. 小要約に複数回出てくる本題
2. 子ども・学習者の困りごと
3. 困りごとの背景要因
4. 支援方法、合理的配慮、本人の希望や合意
5. 検査・診断・相談先など、次に確認すべきこと

見出しごとの書き方:
- 今日の内容: 講座全体で扱ったことを3〜5項目
- 大事なこと: 学習者・先生・保護者が覚えておくべきことを3〜5項目
- わからなかったかもしれないこと: 用語だけを最大10項目
- 次に先生や大人に確認すること: 確認する問いを最大7項目

出力:
## 今日の内容
## 大事なこと
## わからなかったかもしれないこと
## 次に先生や大人に確認すること

小要約:
{summaries}

出力:
/no_think
"""

CORE_TERM_EXTRACTION_PROMPT = """\
以下は学校場面の授業・面談・勉強会・会議などの小要約です。
この回で重要そうな語句・概念だけを抜き出してください。

守ること:
- 入力にある語だけを使う
- 新しい語を作らない
- 人名・学校名・日付・会場案内は除く
- 意味不明な語、要確認語、要確認断片は除く
- 一般語だけの語は除く
- 最大20個
- 1行1語
- 箇条書きだけ

入力:
{summaries}

出力:
/no_think
"""

CLEAN_USER_PROMPT = """\
ASR文字起こしを、読みやすいクリーン文字起こしに整形してください。

目的:
- 面談記録・授業記録テンプレート作成の前処理として使う
- 発言の意味を変えずに読みやすくする

ルール:
1. 原文にない内容を追加しない
2. 推測で補完しない
3. 発言内容を要約しない
4. 決定事項を作らない
5. 相槌、言い直し、明らかな重複は整理してよい
6. 明らかな誤変換は、文脈上確実な場合のみ修正してよい
7. 不明な語は [不明] または [要確認: ...] と書く
8. 時間情報はできるだけ保持する
9. 個人名・学校名などの固有名詞は勝手に修正しない
10. 医療・教育制度に関する語は不確かな場合、要確認にする
11. まとめ・考察・意図の推測を書かない
12. 話者や役割を推測しない。「面接担当」「先生」「生徒」などの役割名を原文にない場合は書かない
13. `**役割**`、`**発言本文**` のようなラベルを作らない

出力形式:
# クリーン文字起こし

## {chunk_time}
発話本文だけを書く

【要確認】
- ...

ASR文字起こし:
{transcript}

出力:
/no_think
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="hybrid_review_transcript.txt or similar")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--prompt_file", default="")
    parser.add_argument("--n_ctx", type=int, default=8192)
    parser.add_argument("--max_chars_per_batch", type=int, default=5000)
    parser.add_argument("--max_batches", type=int, default=0, help="0 = all batches")
    parser.add_argument("--max_tokens_part", type=int, default=1400)
    parser.add_argument("--max_tokens_clean", type=int, default=1600)
    parser.add_argument("--max_tokens_summary", type=int, default=700, help="reserved for compatibility")
    parser.add_argument("--max_tokens_final", type=int, default=1800)
    parser.add_argument("--learner_final_threshold", type=int, default=3, help="Use hierarchical learner final when batch count is at least this value. 0 disables it.")
    parser.add_argument("--learner_final_group_size", type=int, default=3, help="Number of part summaries per intermediate learner summary")
    parser.add_argument("--n_threads", type=int, default=4, help="CPU threads for llama_cpp inference. 4 is safe for 8GB GIGA tablets.")
    parser.add_argument("--n_batch", type=int, default=256, help="Prompt evaluation batch size. Lower values reduce peak RAM on 8GB machines.")
    parser.add_argument("--n_gpu_layers", type=int, default=0, help="Metal/CUDA GPU layers for llama_cpp. 0 = CPU only (Windows default); macOS passes -1 to offload every layer to Metal.")
    parser.add_argument("--max_tokens_learner_summary", type=int, default=900)
    parser.add_argument("--max_tokens_learner_final", type=int, default=1400)
    parser.add_argument("--max_tokens_core_terms", type=int, default=450)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--stage1_only", action="store_true", help="stop after raw/clean transcript and review flags")
    parser.add_argument("--clean_mode", choices=("safe", "llm"), default="safe", help="safe = deterministic clean, llm = Qwen clean with fallback")
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


def clean_llm_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text


def fix_section_heading_bullets(text: str) -> str:
    """Convert known section names output as bullets back to ## headings (Qwen3.5 quirk)."""
    headings = ("英語・専門語・固有名詞", "要確認箇所", "3行サマリー")
    for h in headings:
        text = re.sub(rf"^- {re.escape(h)}\s*$", f"## {h}", text, flags=re.MULTILINE)
    return text


def clean_formatted_output(text: str) -> str:
    """Strip timestamps from 整形記録 bullets, remove template lines, and deduplicate loops."""
    text = fix_section_heading_bullets(text)
    in_seikei = False
    current_heading = ""
    seen_bullets: set[str] = set()
    lines = []
    for line in text.split("\n"):
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            current_heading = heading.group(1)
            in_seikei = current_heading == "整形記録"
            seen_bullets.clear()
        # Remove template description lines copied from prompt (any section)
        if re.match(r"^（.*最大\d+項目.*）$", line.strip()):
            continue
        # Normalize numbered list items so loop removal and section limits work.
        line = re.sub(r"^\s*\d+[.)]\s+", "- ", line)
        if in_seikei:
            # Strip bracketed timestamps: [00:00] and [00:00 --> 00:00]
            line = re.sub(r"\[[\d:\.]+(?:\s*[-–>]+\s*[\d:\.]+)?\]\s*", "", line).rstrip()
            # Strip bare timestamps after optional bullet prefix (Qwen3.5 format)
            # "- 01:05.19 - content"  → "- content"
            # "- 62:08.49〜62:35.89：content" → "- content"
            line = re.sub(r"^(- )?\d{1,3}:\d{2}[.:]\d{2}(?:[〜~]\d{1,3}:\d{2}[.:]\d{2})?[：:]\s*", r"\1", line).rstrip()
            line = re.sub(r"^(- )?\d{1,3}:\d{2}[.:]\d{2}\s+-\s+", r"\1", line).rstrip()
        stripped = line.strip()
        if (
            current_heading in {"英語・専門語・固有名詞", "要確認箇所"}
            and stripped
            and not stripped.startswith("- ")
            and not stripped.startswith("##")
            and not stripped.startswith("（")
        ):
            line = f"- {stripped}"
        # Deduplicate repeated bullet lines (loop artifact)
        stripped = line.strip()
        if stripped.startswith("- ") and stripped in seen_bullets:
            continue
        if stripped.startswith("- "):
            seen_bullets.add(stripped)
        lines.append(line)
    return "\n".join(lines)


_SUMMARY_STOPWORDS = frozenset({
    "chunk", "base", "patch", "empty", "Windows", "AI", "ICT", "PDF",
    "chunk/base", "チャンク", "ベース", "パッチ",
})


def _is_term_list_summary(text: str) -> bool:
    """Return True if ## 3行サマリー looks like a bare term list rather than prose."""
    items = [l.removeprefix("- ").strip() for l in section_lines(text, "3行サマリー") if not is_placeholder_item(l)]
    if not items:
        return True
    first = items[0]
    return first.startswith("主な論点") or len(first) < 20


_TERMS_SECTION_NOISE = frozenset({
    "chunk", "base", "patch", "chunk/base", "lang", "hint", "empty", "AI", "PDF",
    "チャンク", "ベース", "パッチ",
})

# Single lowercase English words that are real technical terms and must NOT be filtered.
_ALLOW_SINGLE_ENGLISH = frozenset({
    "dyslexia", "autism", "mangrove", "ivise", "broca", "wernicke",
})


def normalize_known_terms(text: str) -> str:
    """Fix high-risk known term expansions that small LLMs often hallucinate."""
    replacements = {
        "LD（低機能障害）": "LD（学習障害）",
        "LD(低機能障害)": "LD（学習障害）",
        "ADHD（注意欠如症候群）": "ADHD（注意欠如・多動症）",
        "ADHD(注意欠如症候群)": "ADHD（注意欠如・多動症）",
        "ラーニングスタビティ": "ラーニング・ディサビリティ（学習障害）",
        "ラニングリサビティ": "ラーニング・ディサビリティ（学習障害）",
        "ラーニング・スタビティ": "ラーニング・ディサビリティ（学習障害）",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return filter_terms_section(text)


def _is_whisper_hallucination_term(term: str) -> bool:
    """Return True for single common English words hallucinated by Whisper on Japanese audio."""
    if " " in term or not term:
        return False
    # All-uppercase acronyms (LD, ICT, ADHD, API…) and CamelCase product names are always real.
    if re.match(r'^[A-Z]{2,}$', term):
        return False
    if re.match(r'^[A-Za-z]+[A-Z][a-z]', term):  # has uppercase mid-word → CamelCase
        return False
    # Single pure-ASCII word, all-lowercase (e.g. "listening", "that", "case")
    if re.match(r'^[a-z]{2,12}$', term) and term not in _ALLOW_SINGLE_ENGLISH:
        return True
    # Single Title-case word shorter than 8 chars: likely a hallucination ("Thank", "Here"...)
    # but not a short proper noun with significance.
    if re.match(r'^[A-Z][a-z]{2,6}$', term) and term not in _ALLOW_SINGLE_ENGLISH:
        return True
    return False


def filter_terms_section(text: str) -> str:
    """Remove pipeline-artifact noise words from 英語・専門語・固有名詞."""
    lines = text.split("\n")
    in_terms = False
    result = []
    for line in lines:
        if re.match(r"^##\s+英語・専門語・固有名詞\s*$", line):
            in_terms = True
            result.append(line)
            continue
        if re.match(r"^##\s+", line):
            in_terms = False
        if in_terms and line.strip().startswith("- "):
            term = line.strip().removeprefix("- ").strip()
            if term in _TERMS_SECTION_NOISE or _is_whisper_hallucination_term(term):
                continue
        result.append(line)
    return "\n".join(result)


def split_commajoined_terms(text: str) -> str:
    """Expand bullets like '- A, B, C' in 英語・専門語・固有名詞 into one bullet per term."""
    lines = text.split("\n")
    in_terms = False
    result = []
    for line in lines:
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            in_terms = ("英語・専門語・固有名詞" in heading.group(1))
            result.append(line)
            continue
        if in_terms and line.strip().startswith("- "):
            body = line.strip().removeprefix("- ").strip()
            parts = [p.strip() for p in re.split(r"[,、]", body) if p.strip()]
            if len(parts) > 1:
                result.extend(f"- {p}" for p in parts)
                continue
        result.append(line)
    return "\n".join(result)


def strip_arrow_quote_format(text: str) -> str:
    """Convert '「raw speech」→「summary」と要約' bullets into clean '- summary' bullets.

    The PART LLM sometimes writes a citation-style format that embeds the raw
    utterance. Keep only the summary (the text after the last → arrow).
    """
    lines = text.split("\n")
    result = []
    arrow_pat = re.compile(r'^-\s+「[^」]*」→「([^」]+)」(?:と要約)?$')
    for line in lines:
        m = arrow_pat.match(line.strip())
        if m:
            summary = m.group(1).strip()
            result.append(f"- {summary}")
        else:
            result.append(line)
    return "\n".join(result)


def limit_section_items(text: str, limits: dict[str, int]) -> str:
    current_heading = ""
    counts: dict[str, int] = {}
    lines = []
    for line in text.split("\n"):
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            current_heading = heading.group(1)
            counts[current_heading] = 0
            lines.append(line)
            continue
        if line.strip().startswith("- ") and current_heading in limits:
            counts[current_heading] = counts.get(current_heading, 0) + 1
            if counts[current_heading] > limits[current_heading]:
                continue
        lines.append(line)
    return "\n".join(lines)


def clip_record_item_length(text: str, max_chars: int = 75) -> str:
    """Truncate overlong 整形記録 bullets (LLM copying raw transcript verbatim)."""
    lines = text.split("\n")
    in_seikei = False
    result = []
    for line in lines:
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            in_seikei = (heading.group(1) == "整形記録")
            result.append(line)
            continue
        if in_seikei and line.strip().startswith("- "):
            body = line.strip().removeprefix("- ").strip()
            if len(body) > max_chars:
                body = body[:max_chars].rstrip("。、 ") + "…"
                line = f"- {body}"
        result.append(line)
    return "\n".join(result)


def absorb_preamble_as_record_items(text: str) -> str:
    """Promote free-text summary sentences written before the first ## heading into ## 整形記録 bullets.

    The PART LLM (primed with _ASSISTANT_PREFIX = "## 整形記録\\n") sometimes writes
    concise summary sentences directly under that implicit heading, then repeats the
    heading with verbatim bullets. This captures those summary sentences so they are
    not lost when build_part_summary scans section_lines("整形記録").
    """
    lines = text.split("\n")
    preamble_bullets: list[str] = []
    rest_lines: list[str] = []
    hit_heading = False
    for line in lines:
        if re.match(r"^##\s+", line):
            hit_heading = True
        if not hit_heading:
            body = line.strip()
            if 15 <= len(body) <= 110 and not body.startswith("- ") and not body.startswith("#"):
                preamble_bullets.append(f"- {body}")
                continue
        rest_lines.append(line)
    if not preamble_bullets:
        return text
    existing = section_lines("\n".join(rest_lines), "整形記録")
    seen: set[str] = set()
    merged: list[str] = []
    for item in preamble_bullets + existing:
        key = item.strip()
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return _replace_or_add_section("\n".join(rest_lines), "整形記録", merged)


def _section_needs_fill(text: str, heading: str) -> bool:
    """Return True when heading section has no non-placeholder bullet items."""
    items = [l for l in section_lines(text, heading) if not is_placeholder_item(l)]
    return len(items) == 0


def fill_missing_part_english_terms(result: str, batch: str) -> str:
    """Fill ## 英語・専門語・固有名詞 from batch text when the PART LLM omitted the section.

    On 8GB low-spec machines with max_tokens_part=1400 the model sometimes runs out of
    tokens before generating the terms section. This heuristic scan ensures downstream
    compact summaries always carry terminology for FINAL integration.
    """
    if not _section_needs_fill(result, "英語・専門語・固有名詞"):
        return result
    raw_terms = extract_candidate_terms(batch)
    clean_terms = [
        t for t in raw_terms
        if t.removeprefix("- ").strip() not in _TERMS_SECTION_NOISE
        and not _is_whisper_hallucination_term(t.removeprefix("- ").strip())
    ][:10]
    if not clean_terms:
        return result
    return _replace_or_add_section(result, "英語・専門語・固有名詞", clean_terms)


def clean_transcript_chunk_items(clean_text: str) -> list[tuple[int, str]]:
    """Extract ordered chunk bodies from # クリーン文字起こし."""
    items: list[tuple[int, str]] = []
    chunks = re.split(r"(?=^##\s+chunk_\d+\b.*$)", clean_text, flags=re.MULTILINE)
    for block in chunks:
        header = re.search(r"^##\s+chunk_(\d+)\b.*$", block, flags=re.MULTILINE)
        if not header:
            continue
        body = re.sub(r"^##\s+chunk_\d+\b.*$", "", block, count=1, flags=re.MULTILINE)
        body = clean_asr_fragment(body)
        if not body or is_filler_fragment(body):
            continue
        items.append((int(header.group(1)), body))
    return items


def summarize_general_chunk_body(body: str) -> str:
    """Turn a common interview / visit ASR chunk into a short, source-grounded bullet.

    This intentionally does not attempt topic-specific paraphrasing or fact
    substitution: it only trims the chunk itself, so the output can never
    diverge from what was actually said in the recording.
    """
    return body[:95].rstrip("。、 ") + ("…" if len(body) > 95 else "")


def ensure_temporal_coverage_in_record(text: str, clean_batch: str, max_items: int = 10) -> str:
    """Add concise later-chunk facts when a part summary only covered the beginning."""
    existing = section_lines(text, "整形記録")
    if len(existing) >= max_items:
        return text
    existing_compact = compact_text("\n".join(existing))
    chunks = clean_transcript_chunk_items(clean_batch)
    if len(chunks) <= len(existing) + 2:
        return text

    topic_markers = (
        "消防", "救急", "訓練", "事務", "報告書", "火災", "年間", "少子高齢化",
        "出動", "連携", "支援", "心臓マッサージ", "埋め立て", "リゾート", "研修",
        "講習", "勉強会", "海外", "日本", "ホース", "バックドラフト", "空気",
        "炎", "窒息", "台風", "巡回", "浸水", "防波堤", "地震", "マンホール",
        "勤務", "サイクル", "交流", "見学", "質問", "ポイント", "相談", "支援",
        "検査", "診断", "ICT", "読み書き", "合理的配慮",
    )
    candidates: list[tuple[int, int, str]] = []
    total = max(1, len(chunks))
    for order, (_idx, body) in enumerate(chunks):
        if len(body) < 18:
            continue
        compact = compact_text(body)
        if compact[:24] in existing_compact or any(compact_text(_bullet_body(item))[:24] in compact for item in existing):
            continue
        score = 0
        score += sum(2 for marker in topic_markers if marker in body)
        if re.search(r"\d|[一二三四五六七八九十百千万]+", body):
            score += 2
        if "?" in body or "？" in body:
            score += 1
        if order > total // 3:
            score += 1
        if order > (total * 2) // 3:
            score += 1
        if score <= 0:
            continue
        clipped = summarize_general_chunk_body(body)
        candidates.append((score, order, f"- {clipped}"))

    if not candidates:
        return text
    merged = existing[:]
    seen = {compact_text(_bullet_body(item))[:32] for item in merged}
    for _score, _order, item in sorted(candidates, key=lambda row: (-row[0], row[1])):
        key = compact_text(_bullet_body(item))[:32]
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= max_items:
            break
    if len(merged) == len(existing):
        return text
    return _replace_or_add_section(text, "整形記録", merged)


def strip_reasoning_preamble(text: str) -> str:
    """Strip English/reasoning text that precedes the first ## section heading (FINAL only)."""
    if text.startswith("##"):
        return text
    m = re.search(r"\n##\s", text)
    if m:
        return text[m.start() + 1:]
    # No section headings found — output is pure reasoning, return placeholder
    return "## 整形記録\n（統合処理失敗 — 個別パートの partial_records.md を参照してください）"


def sanitize_unsupported_roles(text: str, source_text: str) -> str:
    """Avoid invented school roles when the source does not mention them."""
    role_markers = ("先生", "教員", "教師", "生徒", "児童", "子ども", "こども")
    if any(marker in source_text for marker in role_markers):
        return text
    replacements = {
        "教員": "話者",
        "教師": "話者",
        "先生": "話者",
        "生徒": "話者",
        "児童": "話者",
        "子ども": "話者",
        "こども": "話者",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def sanitize_unsupported_context_labels(text: str, source_text: str) -> str:
    """Remove common invented context labels copied from generic school prompts."""
    labels = ("授業", "生徒会", "集会", "面談")
    for label in labels:
        if label not in source_text:
            text = re.sub(rf"^.*{re.escape(label)}.*$", "", text, flags=re.MULTILINE)
            text = text.replace(label, "やりとり")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def has_substantive_final(text: str) -> bool:
    bullets = [
        line.strip()
        for line in text.split("\n")
        if line.strip().startswith("- ") and not is_placeholder_item(line)
    ]
    meaningful = [line for line in bullets if len(line.removeprefix("- ").strip()) >= 8]
    return len(meaningful) >= 2


def has_substantive_section(text: str, heading: str, min_items: int = 2) -> bool:
    items = [
        line
        for line in section_lines(text, heading)
        if not is_placeholder_item(line) and len(line.removeprefix("- ").strip()) >= 12
    ]
    return len(items) >= min_items


def section_lines(text: str, heading: str) -> list[str]:
    pattern = rf"^##\s+{re.escape(heading)}\s*$"
    lines = text.split("\n")
    collecting = False
    found: list[str] = []
    for line in lines:
        if re.match(r"^##\s+", line):
            collecting = bool(re.match(pattern, line))
            continue
        if collecting and line.strip().startswith("- "):
            item = line.strip()
            if item not in found:
                found.append(item)
    return found


def is_placeholder_item(line: str) -> bool:
    placeholders = (
        "empty",
        "抽出できませんでした",
        "該当項目は抽出されませんでした",
        "目立つ要確認箇所は抽出されませんでした",
    )
    stripped = line.strip().removeprefix("- ").strip()
    if stripped in ("[ ]", "[]", "なし", "特になし", "該当なし"):
        return True
    return any(word in line for word in placeholders)


_LEARNER_SUPPORT_KEYWORDS = (
    "読み書き",
    "読み",
    "書き",
    "漢字",
    "ノート",
    "宿題",
    "LD",
    "学習障害",
    "読み書き障害",
    "合理的配慮",
    "支援",
    "学校",
    "保護者",
    "子ども",
    "困難",
    "苦手",
    "自己肯定",
    "モチベーション",
    "環境要因",
    "自己肯定感",
    "学習性無力感",
    "視覚認知",
    "眼球運動",
    "音韻",
    "記憶",
    "注意",
    "認知",
    "教材",
    "検査",
    "診断",
    "ICT",
    "iPad",
    "Chromebook",
    "クローンブック",
)

PART_CONTENT_HEADING = "この部分に出てきた内容"
LEGACY_PART_CONTENT_HEADING = "重要事項"
_SAFE_KATAKANA_TERMS = {
    "ケアレスミス",
    "コミュニケーション",
    "リフレーミング",
    "モチベーション",
    "スモールステップ",
    "チェック",
    "イメージ",
    "サイクル",
    "コーディネーター",
    "コーディネータ",
    "スライド",
    "バックドラフト",
    "マンホール",
    "ディスレクシア",
    "アロスミス",
    "コグトレ",
    "ニューロサイエンス",
    "ドリルパーク",
    "ミライシード",
}


def content_section_lines(text: str) -> list[str]:
    return section_lines(text, PART_CONTENT_HEADING) + section_lines(text, LEGACY_PART_CONTENT_HEADING)


def has_learner_support_keyword(text: str) -> bool:
    return any(keyword in text for keyword in _LEARNER_SUPPORT_KEYWORDS)


def has_general_record_topic_keyword(text: str) -> bool:
    markers = (
        "消防", "救急", "火災", "勤務", "訓練", "事務作業", "報告書", "出動",
        "連携", "心臓マッサージ", "研修", "講習", "勉強会", "見学", "交流",
        "埋め立て", "リゾート", "海外", "日本", "ホース", "バックドラフト",
        "バックドラフ", "空気", "炎", "窒息", "台風", "巡回", "浸水",
        "防波堤", "地震", "マンホール", "少子高齢化",
    )
    return any(marker in text for marker in markers)


def is_suspicious_katakana_term(term: str) -> bool:
    body = term.removeprefix("- ").strip()
    if has_learner_support_keyword(body):
        return False
    if body in {"ADHD", "LD", "ICT", "PDF", "iPad", "Chromebook", "SFC", "SC"}:
        return False
    if body.strip("-") in _SAFE_KATAKANA_TERMS:
        return False
    # Long katakana-only fragments are often ASR hallucinations or misheard names.
    return bool(re.fullmatch(r"[ァ-ヴー・ー]{6,}", body))


def is_suspicious_review_term(term: str) -> bool:
    body = term.removeprefix("- ").strip()
    if has_learner_support_keyword(body):
        return False
    if is_suspicious_katakana_term(body):
        return True
    # Misheard proper-name-like phrases should be preserved for review, not
    # promoted as learner-facing terminology.
    if re.search(r"(先生|さん)", body) and re.search(r"[ァ-ヴーA-Za-z]", body):
        return True
    return False


def is_fragment_like_line(line: str) -> bool:
    body = line.removeprefix("- ").strip()
    if has_learner_support_keyword(body):
        return False
    if has_general_record_topic_keyword(body):
        return False
    if len(body) < 16:
        return True
    if body.endswith(("…", "っ", "し", "て", "で", "の", "が", "を", "に")):
        return True
    return False


def split_review_candidates(lines: list[str]) -> tuple[list[str], list[str]]:
    """Move suspicious/noisy candidates out of content and into review checks."""
    kept: list[str] = []
    checks: list[str] = []
    for line in lines:
        if is_placeholder_item(line):
            continue
        body = line.removeprefix("- ").strip()
        if is_suspicious_review_term(body):
            checks.append(f"- 要確認語: {body}")
            continue
        if is_fragment_like_line(body):
            checks.append(f"- 要確認断片: {body}")
            continue
        kept.append(line)
    return kept, checks


def select_fallback_important(lines: list[str], max_items: int = 10) -> list[str]:
    """Pick useful fallback items from the whole long record instead of the first page."""
    candidates: list[tuple[int, int, str]] = []
    fill: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        if is_placeholder_item(line):
            continue
        body = line.removeprefix("- ").strip()
        if len(body) < 18:
            continue
        score = 0
        for keyword in _LEARNER_SUPPORT_KEYWORDS:
            if keyword in body:
                score += 10
        if "..." in body or "…" in body:
            score -= 1
        if score > 0:
            candidates.append((score, idx, line))
        else:
            fill.append((idx, line))

    selected: list[tuple[int, str]] = []
    seen: set[str] = set()
    for _score, idx, line in sorted(candidates, key=lambda item: (-item[0], item[1])):
        key = compact_text(line)[:36]
        if key in seen:
            continue
        seen.add(key)
        selected.append((idx, line))
        if len(selected) >= max_items:
            break

    for idx, line in fill:
        if len(selected) >= max_items:
            break
        key = compact_text(line)[:36]
        if key in seen:
            continue
        seen.add(key)
        selected.append((idx, line))

    return [line for _idx, line in sorted(selected, key=lambda item: item[0])]


def select_fallback_terms(lines: list[str], max_items: int = 15) -> list[str]:
    preferred: list[tuple[int, int, str]] = []
    fill: list[tuple[int, str]] = []
    seen: set[str] = set()
    for idx, line in enumerate(lines):
        if is_placeholder_item(line):
            continue
        term = line.removeprefix("- ").strip()
        if not term or term in _TERMS_SECTION_NOISE or term in seen:
            continue
        if is_suspicious_review_term(term):
            continue
        seen.add(term)
        score = 0
        for keyword in _LEARNER_SUPPORT_KEYWORDS:
            if keyword in term:
                score += 10
        if term in {"LD", "ADHD", "合理的配慮", "読み書き障害", "学習障害", "脳機能", "視覚認知", "眼球運動"}:
            score += 20
        if score > 0:
            preferred.append((score, idx, f"- {term}"))
        else:
            fill.append((idx, f"- {term}"))

    selected = [(idx, line) for _score, idx, line in sorted(preferred, key=lambda item: (-item[0], item[1]))[:max_items]]
    for idx, line in fill:
        if len(selected) >= max_items:
            break
        selected.append((idx, line))
    return [line for _idx, line in sorted(selected, key=lambda item: item[0])]


def fallback_final_record(compact_partial_text: str) -> str:
    important = select_fallback_important(content_section_lines(compact_partial_text), max_items=10)
    terms = select_fallback_terms(section_lines(compact_partial_text, "英語・専門語・固有名詞"), max_items=15)
    checks = [
        line for line in section_lines(compact_partial_text, "要確認箇所")
        if "[00:00]" not in line and not is_placeholder_item(line)
    ][:10]

    def render(items: list[str], placeholder: str) -> str:
        return "\n".join(items) if items else f"- {placeholder}"

    body = "\n".join([
        "## 整形記録",
        render(important, "内容は抽出できませんでした。partial_records.mdを確認してください。"),
        "",
        "## 英語・専門語・固有名詞",
        render(terms, "該当項目は抽出できませんでした。"),
        "",
        "## 要確認箇所",
        render(checks, "目立つ要確認箇所は抽出されませんでした。"),
    ])
    summary_block = build_3line_summary_from_record(body, compact_partial_text)
    return body + "\n\n" + summary_block


def build_fallback_3line_summary(compact_partial_text: str) -> str:
    """Build a fallback ## 3行サマリー block from compact part summaries."""
    important = [line for line in content_section_lines(compact_partial_text) if not is_placeholder_item(line)][:10]
    terms = [line for line in section_lines(compact_partial_text, "英語・専門語・固有名詞") if not is_placeholder_item(line)][:15]
    checks = [
        line for line in section_lines(compact_partial_text, "要確認箇所")
        if "[00:00]" not in line and not is_placeholder_item(line)
    ][:10]
    term_labels = [line.removeprefix("- ").strip() for line in terms[:6]]
    summary_1 = "主な論点: " + "、".join(term_labels) if term_labels else "音声記録の要点を整理しました。"
    summary_2 = f"整形記録は{len(important)}件の内容候補を抽出しています。必要に応じてpart_summaries.mdで前後文を確認してください。"
    summary_3 = f"要確認箇所は{len(checks)}件です。" if checks else "目立つ要確認箇所は抽出されませんでした。"
    return "\n".join([
        "## 3行サマリー",
        f"- {summary_1}",
        f"- {summary_2}",
        f"- {summary_3}",
    ])


def remove_section(text: str, heading: str) -> str:
    pattern = rf"^##\s+{re.escape(heading)}\s*$"
    lines = text.split("\n")
    kept: list[str] = []
    skipping = False
    for line in lines:
        if re.match(r"^##\s+", line):
            skipping = bool(re.match(pattern, line))
            if skipping:
                continue
        if not skipping:
            kept.append(line)
    return "\n".join(kept).rstrip()


def ensure_safe_transcript_note(text: str) -> str:
    note = "\n".join([
        "注意:",
        "このsafe transcriptは、ASR結果をチャンク順・時刻付きで確認しやすく並べた原文確認用の出力です。",
        "正式な確認が必要な場合は raw_transcript.txt および元音声と照合してください。",
    ])
    if "正式な確認が必要な場合は raw_transcript.txt" in text:
        return text.rstrip() + "\n"
    return text.rstrip() + "\n\n" + note + "\n"


def ensure_ai_readable_transcript_note(text: str) -> str:
    note = "\n".join([
        "注意:",
        "このAI整形版は読みやすさを優先した参考出力です。",
        "英語本文の翻訳、語句の補正、意味の変化が含まれる可能性があります。",
        "正式な確認には safe_transcript.md、raw_transcript.txt、および元音声を使用してください。",
    ])
    if "このAI整形版は読みやすさを優先した参考出力です。" in text:
        return text.rstrip() + "\n"
    return text.rstrip() + "\n\n" + note + "\n"


def ensure_clean_transcript_note(text: str) -> str:
    """Compatibility wrapper for older callers."""
    return ensure_safe_transcript_note(text)


def sanitize_clean_transcript(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in {"発話本文", "発話本文だけを書く", "ASR文字起こし:", "出力:"}:
            continue
        if stripped == "/no_think":
            continue
        if re.match(r"^\*\*役割\*\*", stripped):
            continue
        if "文脈から推測" in stripped:
            continue
        line = re.sub(r"^\*\*発言本文\*\*\s*[:：]\s*", "", line)
        lines.append(line.rstrip())
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"、。", "。", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_transcript_has_inference(text: str) -> bool:
    forbidden = (
        "インタビュアー",
        "面接担当",
        "文脈から推測",
        "どのような配慮",
        "どのようなサポート",
        "この方、",
        "質問:",
        "質問：",
    )
    return any(word in text for word in forbidden)


def deterministic_clean_transcript(batch: str) -> str:
    md, _txt, _count = build_formatted_transcript(batch)
    md = md.replace("# 整形文字起こし", "# クリーン文字起こし", 1)
    return sanitize_clean_transcript(md)


def ensure_meeting_record_note(text: str) -> str:
    note = "\n".join([
        "注意:",
        "この記録は、文字起こしとAI整形をもとにした確認用メモです。",
        "正式記録として使用する場合は、原文文字起こしおよび元音声と照合してください。",
    ])
    if "正式記録として使用する場合は、原文文字起こし" in text:
        return text.rstrip() + "\n"
    return text.rstrip() + "\n\n" + note + "\n"


def mark_record_for_reprocess(text: str, reason: str) -> str:
    """Add a visible warning when the final integrated record used deterministic fallback."""
    if "## 要再処理" in text:
        return text
    warning = "\n".join([
        "## 要再処理",
        f"- {reason}",
        "- 学習者向けに使う前に、partial_records.md、part_summaries.md、raw_transcript.txtを確認してください。",
        "",
    ])
    return warning + text.lstrip()


def build_review_flags(clean_text: str, final_record: str) -> str:
    flags: list[str] = []
    if "## 要再処理" in final_record:
        flags.append("- FINAL整形でfallbackを使用。学習者向けに使う前に再処理または人手確認が必要です。")
    combined_text = clean_text + "\n" + final_record
    if "ディスレクチュア" in combined_text or "セレクチュア" in combined_text:
        flags.append("- `ディスレクチュア/セレクチュア` は `ディスレクシア` の誤認識の可能性があります。")
    if "SOC" in combined_text:
        flags.append("- `SOC` は文脈上 `SFC` の誤認識の可能性があります。")
    for line in clean_text.splitlines():
        stripped = line.strip()
        if "[要確認" in stripped or "[不明" in stripped or stripped.startswith("- [要確認"):
            flags.append(f"- {stripped.lstrip('- ').strip()}")
    clean_blocks = re.split(r"(?=^##\s+chunk_\d+\s+.+$)", clean_text, flags=re.MULTILINE)
    for block in clean_blocks:
        heading = re.search(r"^##\s+(chunk_\d+\s+.+)$", block, flags=re.MULTILINE)
        if not heading:
            continue
        if (
            "認識: fw_patch" in block
            or "認識: ja_base + fw_patch" in block
            or "認識: parakeet_patch" in block
            or "認識: ja_base + parakeet_patch" in block
        ):
            flags.append(f"- [{heading.group(1).strip()}] 英語補助認識を使用。必要に応じて元音声と照合してください。")
    for item in section_lines(final_record, "要確認箇所"):
        if not is_placeholder_item(item):
            flags.append(item)
    for item in section_lines(final_record, "わからなかったかもしれないこと"):
        if not is_placeholder_item(item):
            body = item.removeprefix("- ").strip()
            if is_suspicious_review_term(body) or "不明" in body or "正確" in body:
                flags.append(f"- 要確認語: {body}")
    for item in section_lines(final_record, "次に先生や大人に確認すること"):
        body = item.removeprefix("- ").strip()
        if (
            not is_placeholder_item(item)
            and ("不明" in body or "正しい表記" in body or "正確な表記" in body or "元音声" in body)
        ):
            flags.append(item)

    unique: list[str] = []
    seen: set[str] = set()
    for flag in flags:
        key = re.sub(r"\s+", "", flag)
        if key and key not in seen:
            seen.add(key)
            unique.append(flag)

    if not unique:
        unique = ["- 目立つ要確認箇所はありません。"]
    return "# 要確認リスト\n\n" + "\n".join(unique) + "\n"


def render_section_only(text: str, heading: str) -> str:
    """Render one markdown section and its bullet items."""
    items = section_lines(text, heading)
    if not items:
        return f"## {heading}\n"
    return "\n".join([f"## {heading}"] + items)


LEARNER_REQUIRED_HEADINGS = (
    "今日の内容",
    "大事なこと",
    "わからなかったかもしれないこと",
    "次に先生や大人に確認すること",
)
LEARNER_EMPTY_SECTION_PLACEHOLDERS = {
    "今日の内容": "- 内容を十分に抽出できませんでした。",
    "大事なこと": "- 大事なことは十分に抽出できませんでした。",
    "わからなかったかもしれないこと": "- 目立つ項目はありません。",
    "次に先生や大人に確認すること": "- この記録を学習に使う前に、先生や大人と内容を確認する。",
}
GENERIC_REVIEW_CHECK = "- 聞き取りが不確かな語句や断片があるため、元音声で確認する。"

CORE_TOPIC_TERMS = (
    "読み書き障害",
    "ICT支援",
    "読み書き",
    "読み書き困難",
    "困難",
    "LD",
    "学習障害",
    "合理的配慮",
    "氷山モデル",
    "視覚認知",
    "眼球運動",
    "音韻",
    "音韻認識",
    "語彙",
    "記憶",
    "注意",
    "注意集中",
    "モチベーション",
    "自己肯定",
    "自己肯定感",
    "自己効力感",
    "学習性無力感",
    "近接発達領域",
    "本人の合意",
    "環境要因",
    "教材",
    "支援",
    "検査",
    "診断",
    "スモールステップ",
)
CORE_TOPIC_ALIASES = {
    "読み書き困難": ("読み書き困難", "読み書きに困難", "読み書き障害", "読み書きの困難"),
    "読み書き障害": ("読み書き障害", "読み書き困難", "読み書きに困難"),
    "学習障害": ("学習障害", "LD"),
    "LD": ("LD", "学習障害"),
    "音韻認識": ("音韻認識", "音韻"),
    "音韻": ("音韻", "音韻認識"),
    "自己肯定感": ("自己肯定感", "自己肯定"),
    "自己肯定": ("自己肯定", "自己肯定感"),
    "環境要因": ("環境要因", "環境"),
    "注意集中": ("注意集中", "注意"),
    "合理的配慮": ("合理的配慮", "配慮"),
}


def has_learner_sections(text: str) -> bool:
    return any(section_lines(text, heading) for heading in LEARNER_REQUIRED_HEADINGS)


def learner_format_check_passed(text: str) -> bool:
    return all(f"## {heading}" in text for heading in LEARNER_REQUIRED_HEADINGS)


def fill_empty_learner_sections(text: str) -> str:
    result = text
    for heading in LEARNER_REQUIRED_HEADINGS:
        if not section_lines(result, heading):
            result = _replace_or_add_section(result, heading, [LEARNER_EMPTY_SECTION_PLACEHOLDERS[heading]])
    return result


def _bullet_body(line: str) -> str:
    return line.strip().removeprefix("- ").strip()


def is_long_noise_fragment(text: str) -> bool:
    body = _bullet_body(text)
    if has_learner_support_keyword(body):
        return False
    if has_general_record_topic_keyword(body):
        return False
    if len(body) >= 55 and ("…" in body or re.search(r"(っていう|なんですけど|ということで|ちょっと)", body)):
        return True
    if len(body) >= 75 and not re.search(r"[。.!?！？]$", body):
        return True
    return False


def compact_review_item(line: str) -> str:
    body = _bullet_body(line)
    body = re.sub(r"^(要確認語|要確認断片)\s*[:：]\s*", "", body).strip()
    body = body.strip("「」")
    if not body or is_long_noise_fragment(body):
        return GENERIC_REVIEW_CHECK
    if len(body) > 28:
        return GENERIC_REVIEW_CHECK
    return f"- 「{body}」の意味や正確な表記を確認する。"


def scrub_review_noise_for_llm(text: str) -> str:
    """Keep review signals, but do not pass raw noisy ASR fragments to final LLM."""
    lines: list[str] = []
    generic_added = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            lines.append(line)
            continue
        body = _bullet_body(stripped)
        if body.startswith(("要確認語:", "要確認語：", "要確認断片:", "要確認断片：")) or is_long_noise_fragment(body):
            compacted = compact_review_item(stripped)
            if compacted == GENERIC_REVIEW_CHECK:
                if generic_added:
                    continue
                generic_added = True
            lines.append(compacted)
            continue
        lines.append(line)
    return "\n".join(lines)


def is_ai_overclaim_line(line: str) -> bool:
    body = _bullet_body(line)
    return bool(re.search(r"AI[がは].*(判断|指導|解決|把握|評価|助言|行う|行った)", body))


def sanitize_learner_noise(text: str) -> str:
    """Remove noisy final bullets and compress raw review fragments after LLM generation."""
    result_lines: list[str] = []
    current_heading = ""
    generic_review_added = False
    for line in text.splitlines():
        heading_match = re.match(r"^##\s+(.+?)\s*$", line.strip())
        if heading_match:
            current_heading = heading_match.group(1)
            result_lines.append(line)
            continue
        stripped = line.strip()
        if not stripped.startswith("- "):
            result_lines.append(line)
            continue

        body = _bullet_body(stripped)
        noisy_review = (
            body.startswith(("要確認語:", "要確認語：", "要確認断片:", "要確認断片："))
            or is_long_noise_fragment(body)
            or is_ai_overclaim_line(body)
        )
        if noisy_review and current_heading in {"今日の内容", "大事なこと"}:
            continue
        if noisy_review and current_heading in {"わからなかったかもしれないこと", "次に先生や大人に確認すること"}:
            compacted = compact_review_item(stripped)
            if compacted == GENERIC_REVIEW_CHECK:
                if generic_review_added:
                    continue
                generic_review_added = True
            result_lines.append(compacted)
            continue
        result_lines.append(line)
    return "\n".join(result_lines)


def detect_learning_record_context(text: str) -> dict:
    """Detect inputs that are likely outside school / learning-support records."""
    compact = compact_text(text)
    positive_terms = (
        "学校", "授業", "講義", "講座", "面談", "勉強会", "会議", "研修", "インタビュー",
        "見学", "交流", "質問", "学習", "支援", "教育", "先生", "保護者", "生徒",
        "児童", "子ども", "こども", "教材", "通級", "教育センター", "合理的配慮",
        "ICT", "訓練", "報告書", "発表", "相談",
    )
    medical_terms = (
        "病院", "患者", "診察", "体温", "血圧", "CT", "麻酔", "縫合", "薬局",
        "保険会社", "怪我", "けが", "傷", "打撲", "あご", "意識障害", "けいれん",
        "コンクッション", "呼吸困難", "心拍", "全身麻酔", "医者", "看護",
    )
    travel_noise_terms = ("ディズニー", "Disneyland", "ホテル", "飛行機", "タクシー")
    positive_hits = [term for term in positive_terms if term in compact]
    medical_hits = [term for term in medical_terms if term in compact]
    travel_hits = [term for term in travel_noise_terms if term in compact]
    outside = len(medical_hits) >= 4 and len(positive_hits) <= 2
    if len(medical_hits) >= 6 and travel_hits:
        outside = True
    return {
        "passed": not outside,
        "reason": "" if not outside else "outside_learning_record_context",
        "positive_hits": positive_hits[:10],
        "outside_hits": (medical_hits + travel_hits)[:12],
    }


def learner_content_quality_check(source_text: str, final_text: str, dynamic_terms: list[str] | None = None) -> dict:
    """Detect a formally valid but topic-drifting learner final."""
    def aliases(term: str) -> tuple[str, ...]:
        if term in CORE_TOPIC_ALIASES:
            return CORE_TOPIC_ALIASES[term]
        dynamic_aliases = [term]
        for key in ("読み書き", "読み", "漢字", "視覚認知", "眼球運動", "音韻", "語彙", "記憶", "注意", "認知", "教材", "特性", "検査", "診断", "支援"):
            if key in term and key not in dynamic_aliases:
                dynamic_aliases.append(key)
        return tuple(dynamic_aliases)

    dynamic_terms = [term for term in (dynamic_terms or []) if term and term in source_text]
    context_check = detect_learning_record_context(source_text)
    if not context_check["passed"]:
        return {
            "passed": False,
            "reason": context_check["reason"],
            "missing_core_terms": [],
            "context_check": context_check,
        }
    if not dynamic_terms:
        return {
            "passed": False,
            "reason": "no_dynamic_core_terms",
            "missing_core_terms": [],
            "context_check": context_check,
        }
    source_terms = list(dict.fromkeys(dynamic_terms))
    dynamic_alias_text = " ".join(dynamic_terms)

    def occurrence_count(term: str) -> int:
        return max(source_text.count(alias) for alias in aliases(term))

    for term in CORE_TOPIC_TERMS:
        if term in source_terms:
            continue
        if not any(alias in source_text for alias in aliases(term)):
            continue
        if dynamic_terms:
            # When Qwen extracted recording-specific core terms, do not let a
            # passing mention of a fixed dyslexia keyword force a fallback for
            # unrelated lessons or support meetings.
            if any(alias in dynamic_alias_text for alias in aliases(term)) or occurrence_count(term) >= 4:
                source_terms.append(term)
        else:
            source_terms.append(term)
    for term in dynamic_terms or []:
        if term not in source_terms and term in source_text:
            source_terms.append(term)
    missing = [term for term in source_terms if not any(alias in final_text for alias in aliases(term))]
    general_record_context = (
        has_general_record_topic_keyword(source_text)
        and not any(term in source_text for term in ("読み書き", "学習障害", "合理的配慮", "不登校", "緘黙", "通級", "学習支援室"))
    )
    if general_record_context and source_terms:
        allowed_missing = max(2, len(source_terms) // 2)
        passed = len(missing) <= allowed_missing and len(section_lines(final_text, "今日の内容")) >= 3
        return {
            "passed": passed,
            "reason": "" if passed else "missing_core_topic_terms",
            "missing_core_terms": missing if not passed else [],
            "context_check": context_check,
        }
    if len(source_terms) < 2:
        passed = True
    else:
        # In learner-support records, dropping a detected core topic should route
        # to deterministic merge instead of silently accepting a fluent but thin final.
        passed = len(missing) == 0
    return {
        "passed": passed,
        "reason": "" if passed else "missing_core_topic_terms",
        "missing_core_terms": missing if not passed else [],
        "context_check": context_check,
    }


def learner_default_quality_metadata() -> dict:
    return {
        "format_check_passed": False,
        "content_quality_check_passed": False,
        "quality_gate_reason": "",
        "missing_core_terms": [],
        "final_generation_mode": "",
        "dynamic_core_terms": [],
        "dynamic_core_terms_count": 0,
        "next_checks_completed": False,
        "next_checks_count": 0,
        "target_context_check_passed": True,
        "target_context_reason": "",
        "target_context_positive_hits": [],
        "target_context_outside_hits": [],
    }


def clean_core_term(term: str) -> str:
    term = term.strip().removeprefix("- ").strip()
    term = re.sub(r"^\d+[\.)]\s*", "", term).strip()
    term = term.strip("「」『』[]（）()、。:： ")
    term = re.sub(r"\s+", "", term)
    return term


def sanitize_dynamic_core_terms(raw_text: str, source_text: str, max_terms: int = 20) -> list[str]:
    source_compact = compact_text(scrub_review_noise_for_llm(source_text))
    review_terms = {
        clean_core_term(line)
        for line in section_lines(source_text, "要確認箇所")
        if _bullet_body(line).startswith(("要確認語:", "要確認語：", "要確認断片:", "要確認断片："))
    }
    generic_noise = {
        "今日の内容", "大事なこと", "要確認箇所", "確認事項", "授業", "講座", "説明", "内容",
        "先生", "子ども", "学校", "保護者", "学習", "支援", "必要", "重要", "確認",
        "チェック", "イメージ", "メリット", "シンプル", "チャンス", "アプローチ",
        "コンサート", "フェンス", "ヒットテロ", "パズル", "アプリ利用", "生きる力",
        "巡回指導", "勤務経験", "学童・女学校", "コーディネーター", "学年主", "経理処理",
        "カヌー競技", "障害者", "共に取り組む", "自信や楽しさ", "モチベーション源",
        "ストレス", "メカニズム",
    }
    short_allowed = set(CORE_TOPIC_TERMS) | {"漢字", "語彙", "記憶", "注意", "音韻", "検査", "診断"}
    scored_terms: list[tuple[int, int, str]] = []
    raw_units: list[str] = []
    for line in raw_text.splitlines():
        raw_units.extend(part.strip() for part in re.split(r"[,、，]", line) if part.strip())
    for line in raw_units:
        if not line.strip():
            continue
        term = clean_core_term(line)
        if not term:
            continue
        if term in {"守ること", "入力", "出力"}:
            continue
        if term.startswith(("要確認語", "要確認断片")):
            continue
        if term in generic_noise or term in review_terms:
            continue
        if len(term) < 3 and term not in short_allowed:
            continue
        if len(term) > 24:
            continue
        if term.endswith(("先生", "さん")):
            continue
        if is_suspicious_review_term(term) or is_long_noise_fragment(term):
            continue
        if term not in source_compact:
            continue
        if re.fullmatch(r"[ぁ-んー]{3,}", term):
            continue
        source_count = source_compact.count(term)
        score = 0
        if term in CORE_TOPIC_TERMS:
            score += 5
        if has_learner_support_keyword(term):
            score += 3
        if re.search(r"[一-龥]", term):
            score += 1
        if source_count >= 2:
            score += 2
        if re.search(r"(消防|救急|火災|勤務|訓練|事務|報告書|出動|連携|研修|講習|勉強会|交流|見学|質問|ポイント|埋め立て|リゾート|海外|日本|ホース|バックドラフト|台風|巡回|浸水|防波堤|地震|マンホール|少子高齢化)", term):
            score += 3
        if re.search(r"(困難|障害|認識|認知|記憶|配慮|無力感|肯定感|効力感|検査|診断|音韻|語彙|注意|モチベーション)", term):
            score += 3
        if re.search(r"(経験|指導|処理|競技|学年|学校|先生|小平|女学校)", term):
            score -= 3
        if score < 3:
            continue
        if term not in [item[2] for item in scored_terms]:
            scored_terms.append((score, len(scored_terms), term))
    return [term for _score, _idx, term in sorted(scored_terms, key=lambda item: (-item[0], item[1]))[:max_terms]]


def build_core_term_extraction_source(source_text: str) -> str:
    """Use content-bearing sections for dynamic term extraction, avoiding noisy term lists."""
    lines: list[str] = []
    for heading in (PART_CONTENT_HEADING, "今日の内容", "大事なこと", "整形記録"):
        lines.extend(section_lines(source_text, heading))
    if not lines:
        lines = [
            line.strip()
            for line in source_text.splitlines()
            if line.strip().startswith("- ") and "要確認" not in line
        ]
    return "\n".join(lines)


def extract_dynamic_core_terms(llm, source_text: str, max_tokens: int, log_path: Path, output_dir: Path) -> list[str]:
    start = time.perf_counter()
    prompt_source = scrub_review_noise_for_llm(build_core_term_extraction_source(source_text))
    raw = call_llm_core_terms(
        llm,
        CORE_TERM_EXTRACTION_PROMPT.format(summaries=prompt_source[:18000]),
        max_tokens,
    )
    terms = sanitize_dynamic_core_terms(raw, source_text)
    seconds = time.perf_counter() - start
    append_log(log_path, f"dynamic_core_terms_seconds={seconds:.3f} count={len(terms)} terms={','.join(terms)}")
    (output_dir / "dynamic_core_terms_raw.md").write_text(raw.strip() + "\n", encoding="utf-8")
    (output_dir / "dynamic_core_terms.md").write_text(
        "\n".join(f"- {term}" for term in terms) + ("\n" if terms else ""),
        encoding="utf-8",
    )
    return terms


def build_learner_quality_gate_report(metadata: dict) -> str:
    ok = (
        metadata.get("format_check_passed") is True
        and metadata.get("content_quality_check_passed") is True
        and not metadata.get("missing_core_terms")
        and metadata.get("target_context_check_passed") is not False
        and metadata.get("dynamic_core_terms_count", 0) > 0
    )
    lines = [
        "## 品質ゲート結果",
        "",
        "**OK**" if ok else "**要確認**",
        "",
        f"- format_check_passed: {str(metadata.get('format_check_passed')).lower()}",
        f"- content_quality_check_passed: {str(metadata.get('content_quality_check_passed')).lower()}",
        f"- final_generation_mode: {metadata.get('final_generation_mode') or '未実行'}",
    ]
    reason = metadata.get("quality_gate_reason") or ""
    missing = metadata.get("missing_core_terms") or []
    dynamic_terms = metadata.get("dynamic_core_terms") or []
    next_checks_completed = bool(metadata.get("next_checks_completed"))
    next_checks_count = metadata.get("next_checks_count")
    target_context_passed = metadata.get("target_context_check_passed")
    target_context_reason = metadata.get("target_context_reason") or ""
    if reason:
        lines.append(f"- quality_gate_reason: {reason}")
    if target_context_passed is not None:
        lines.append(f"- target_context_check_passed: {str(target_context_passed).lower()}")
    if target_context_reason:
        lines.append(f"- target_context_reason: {target_context_reason}")
    if missing:
        lines.append("- missing_core_terms: " + "、".join(missing))
    if dynamic_terms:
        lines.append("- dynamic_core_terms: " + "、".join(dynamic_terms))
    else:
        lines.append("- dynamic_core_terms: なし")
    if next_checks_count:
        lines.append(f"- next_checks_count: {next_checks_count}")
    if next_checks_completed:
        lines.append("- next_checks_completed: true")
    if ok:
        lines.extend([
            "- 形式が成立している。",
            "- 本題語が最終要約に残っている。",
            "- 品質ゲートによる決定的統合への切替は不要。",
        ])
    else:
        lines.extend([
            "- 形式または本題語保持に問題がある可能性がある。",
            "- 学習者向けに使う前に、part_summaries.md と元音声を確認する。",
        ])
    return "\n".join(lines) + "\n"


def clean_learner_summary(text: str) -> str:
    text = clean_formatted_output(strip_reasoning_preamble(text))
    text = normalize_known_terms(text)
    text = sanitize_unsupported_roles(text, text)
    text = sanitize_unsupported_context_labels(text, text)
    text = sanitize_learner_noise(text)
    text = limit_section_items(text, {
        "今日の内容": 5,
        "大事なこと": 5,
        "わからなかったかもしれないこと": 10,
        "次に先生や大人に確認すること": 7,
    })
    text = sanitize_learner_noise(text)
    text = sanitize_important_term_list(text)
    text = fill_empty_learner_sections(text)
    return text.strip()


def sanitize_important_term_list(text: str) -> str:
    """Move glossary-like bullets out of 大事なこと."""
    important = section_lines(text, "大事なこと")
    if not important:
        return text
    kept: list[str] = []
    moved_terms: list[str] = []
    for item in important:
        body = item.removeprefix("- ").strip()
        term_match = re.match(r"^\*\*([^*]{1,24})\*\*\s*[:：]", body) or re.match(r"^([^:：。]{1,18})\s*[:：]", body)
        glossary_like = bool(term_match) and not re.search(r"(する|した|できる|必要|大切|重要|確認|困|学|支え|扱)", body)
        if glossary_like:
            term = term_match.group(1).strip("「」『』 ")
            if term and term not in moved_terms:
                moved_terms.append(term)
            continue
        kept.append(item)
    if not moved_terms:
        return text

    result = _replace_or_add_section(
        text,
        "大事なこと",
        kept or ["- 大事なことは、今日の内容と元音声を確認して整理する必要があります。"],
    )
    unknown = [
        line.removeprefix("- ").strip()
        for line in section_lines(result, "わからなかったかもしれないこと")
        if not is_placeholder_item(line)
    ]
    for term in moved_terms:
        if term not in unknown:
            unknown.append(term)
    result = _replace_or_add_section(
        result,
        "わからなかったかもしれないこと",
        [f"- {term}" for term in unknown[:10]] or ["- 目立つ項目はありません。"],
    )
    return result


def build_adaptive_next_checks(text: str, dynamic_terms: list[str] | None = None) -> list[str]:
    """Create learner-support follow-up questions from source topics."""
    source_body = compact_text(text)
    dynamic_body = " ".join(dynamic_terms or [])

    def has_any(*terms: str) -> bool:
        return any(term in source_body for term in terms)

    def score(candidates: list[str]) -> int:
        total = 0
        for term in candidates:
            if term in source_body:
                total += 1
            if term in dynamic_body:
                total += 3
        return total

    reading_score = score(["読み書き困難", "読み書き障害", "読み書き", "漢字", "音韻", "視覚認知"])
    ict_score = score(["ICT", "ICTツール", "デジタル教科書", "iPad", "Chromebook", "オンライン", "チャット", "AI", "プロンプト"])
    support_room_score = score(["学習支援室", "通級", "個別指導", "支援室", "教育センター"])
    attendance_score = score(["不登校", "場面緘黙", "緘黙", "オンライン", "チャット"])
    diagnosis_score = score(["LD", "ADHD", "学習障害", "診断", "検査", "アセスメント"])
    reading_dynamic = any(term in dynamic_body for term in ("読み書き困難", "読み書き障害", "読み書き", "漢字", "音韻", "視覚認知"))

    reading_focus = reading_score >= 2 and (reading_dynamic or reading_score >= max(ict_score, support_room_score, attendance_score, 4))
    ict_focus = ict_score > 0
    support_room_focus = support_room_score > 0
    attendance_focus = attendance_score > 0
    diagnosis_focus = diagnosis_score > 0

    base_checks: list[str] = [
        "- 学習者・子ども・参加者は、どの場面で困っている、または確認したいと感じているか。",
        "- その困りごとは、環境、教材、進め方、関わる人、本人の特性のどれと関係していそうか。",
        "- 何を変えると、少し取り組みやすくなるのか。",
        "- 本人や関係者が「これならできそう」と思える方法は何か。",
        "- 次に誰が、何を、いつまでに確認するのか。",
    ]
    focus_checks: list[str] = []
    if reading_focus:
        focus_checks.append("- 読む・書く・聞く・見る・覚えることのどこで困っているのか。")
    if ict_focus:
        focus_checks.append("- どのICTツールや入力方法なら、本人が学習や表現に使いやすいか。")
    if attendance_focus:
        focus_checks.append("- オンライン参加、チャット、録画など、登校や発話が難しい時の参加方法を選べるか。")
    if support_room_focus:
        focus_checks.append("- 学習支援室、通級、教育センターなど、どこに相談や個別支援の窓口があるか。")
    if diagnosis_focus:
        focus_checks.append("- 検査を受ける場合、何のために受けるのかを本人に説明できているか。")
    if has_any("支援", "相談", "学校") and not support_room_focus:
        focus_checks.append("- 学校や関係機関に、学習や学校生活の困りごとを相談できる仕組みがあるか。")
    return focus_checks + base_checks


def complete_learner_next_checks(
    final_text: str,
    source_text: str,
    dynamic_terms: list[str] | None = None,
    min_items: int = 5,
    max_items: int = 7,
) -> tuple[str, bool, int]:
    """Fill thin learner follow-up questions without changing other sections."""
    heading = "次に先生や大人に確認すること"
    existing = [
        line for line in section_lines(final_text, heading)
        if not is_placeholder_item(line)
    ]
    if len(existing) >= min_items:
        return final_text, False, len(existing)

    candidates = build_adaptive_next_checks(source_text + "\n\n" + final_text, dynamic_terms)
    merged: list[str] = []
    seen: set[str] = set()
    for item in existing + candidates:
        body = _bullet_body(item)
        key = compact_text(body)
        if not body or key in seen:
            continue
        seen.add(key)
        merged.append(item if item.strip().startswith("- ") else f"- {body}")
        if len(merged) >= max_items:
            break

    if len(merged) <= len(existing):
        return final_text, False, len(existing)
    return _replace_or_add_section(final_text, heading, merged), True, len(merged)


def build_deterministic_learner_summary(text: str, dynamic_terms: list[str] | None = None) -> str:
    important_source = select_fallback_important(
        section_lines(text, "今日の内容")
        + section_lines(text, "大事なこと")
        + content_section_lines(text)
        + section_lines(text, "整形記録"),
        max_items=5,
    )
    term_source = select_fallback_terms(
        section_lines(text, "英語・専門語・固有名詞")
        + section_lines(text, "大事なこと"),
        max_items=12,
    )
    source_body = compact_text(text)
    dynamic_terms = [
        term for term in (dynamic_terms or [])
        if term and term in source_body and not is_suspicious_review_term(term)
    ]

    def has_any(*terms: str) -> bool:
        return any(term in source_body for term in terms)

    def present_terms(candidates: list[str]) -> list[str]:
        found: list[str] = []
        for term in candidates:
            if term in source_body and term not in found:
                found.append(term)
        return found

    fixed_topic_terms = present_terms([
        "読み書き困難", "読み書き障害", "読み書き", "氷山モデル", "視覚認知", "眼球運動",
        "音韻認識", "音韻", "語彙", "記憶", "注意", "モチベーション", "学習性無力感",
        "自己肯定感", "自己効力感", "近接発達領域", "合理的配慮", "検査", "診断",
        "スモールステップ", "ICT", "ICTツール", "デジタル教科書", "学習支援室", "通級",
        "不登校", "場面緘黙", "オンライン", "チャット", "個別指導", "LD", "ADHD",
        "アセスメント", "AI", "プロンプト",
    ])
    topic_terms: list[str] = []
    for term in dynamic_terms + fixed_topic_terms:
        if term not in topic_terms:
            topic_terms.append(term)

    dynamic_body = " ".join(dynamic_terms)

    def focus_score(candidates: list[str], *, dynamic_weight: int = 3) -> int:
        score = 0
        for term in candidates:
            if term in source_body:
                score += 1
            if term in dynamic_body:
                score += dynamic_weight
        return score

    reading_score = focus_score(["読み書き困難", "読み書き障害", "読み書き", "漢字", "音韻", "視覚認知"])
    ict_score = focus_score(["ICT", "ICTツール", "デジタル教科書", "iPad", "Chromebook", "オンライン", "チャット", "AI", "プロンプト"])
    support_room_score = focus_score(["学習支援室", "通級", "個別指導", "支援室", "教育センター"])
    attendance_score = focus_score(["不登校", "場面緘黙", "緘黙", "オンライン", "チャット"])
    diagnosis_score = focus_score(["LD", "ADHD", "学習障害", "診断", "検査", "アセスメント"])
    reading_dynamic = any(term in dynamic_body for term in ("読み書き困難", "読み書き障害", "読み書き", "漢字", "音韻", "視覚認知"))

    reading_focus = reading_score >= 2 and (reading_dynamic or reading_score >= max(ict_score, support_room_score, attendance_score, 4))
    ict_focus = ict_score > 0
    support_room_focus = support_room_score > 0
    attendance_focus = attendance_score > 0
    diagnosis_focus = diagnosis_score > 0

    today: list[str] = []
    if reading_focus:
        today.append("- 読み書きに困難のある子どもについて、表に見える行動だけでなく背景にある要因を理解する内容だった。")
    else:
        focus_words = topic_terms[:4] or present_terms(["学習", "学校生活", "支援", "相談", "ICT"])
        if focus_words:
            today.append("- " + "、".join(focus_words) + "を中心に、記録全体の内容を整理した。")
    learner_trouble_context = has_any("困難", "苦手", "学習障害", "LD", "ADHD", "不登校", "緘黙", "読み書き", "宿題", "ノート")
    trouble_terms = present_terms(["読む", "書く", "漢字", "ノート", "宿題", "聞く", "見る", "覚える", "話す", "参加", "登校"]) if learner_trouble_context else []
    if trouble_terms:
        today.append("- 扱われた困りごとは、" + "、".join(trouble_terms[:6]) + "に関係していた。")
    if topic_terms:
        today.append("- 主な内容は、" + "、".join(topic_terms[:12]) + "。")
    if ict_focus:
        today.append("- ICTツール、オンライン参加、チャット、AIなどを、子どもの学びや表現を助ける手段として扱った。")
    if support_room_focus:
        today.append("- 学習支援室や通級など、学校内外での個別的な支援のあり方について話した。")
    for item in important_source:
        if len(today) >= 5:
            break
        if item not in today:
            today.append(item)

    important: list[str] = []
    if reading_focus and has_any("努力不足", "困難", "苦手", "読み書き"):
        important.append("- 読み書きの困難は、本人の努力不足だけで説明してはいけない。")
    elif has_any("困難", "苦手", "学習障害", "LD", "ADHD", "不登校", "緘黙"):
        important.append("- 子どもの困りごとは本人の努力不足だけでなく、環境、教材、支援方法、本人の特性などと合わせて考える。")
    hidden_terms = present_terms(["環境", "教材", "先生", "視覚認知", "聞く", "記憶", "注意", "モチベーション", "体", "ICT", "オンライン", "チャット"])
    if hidden_terms and has_any("困難", "苦手", "学習障害", "LD", "ADHD", "不登校", "緘黙", "読み書き"):
        important.append("- 見える行動の下には、" + "、".join(hidden_terms[:9]) + "などの要因がある。")
    if has_any("読めて", "語彙", "逐次読み", "読み飛ば"):
        important.append("- 読めているように見える子でも、語彙や経験を頼りに大まかに読んでいるだけの場合がある。")
    if reading_focus and has_any("漢字", "ノート", "スモールステップ", "自己肯定", "モチベーション"):
        important.append("- 漢字やノートを何度も書かせるだけでなく、子どもに合う方法を探し、スモールステップで取り組める入口を作ることが重要。")
    if ict_focus:
        important.append("- ICTやオンラインの方法は、読みにくさ、書きにくさ、話しにくさがある子どもの参加や表現を助ける選択肢になる。")
    if support_room_focus or has_any("自己肯定", "モチベーション"):
        important.append("- 支援はできない部分を目立たせるためではなく、本人が安心して学び、自分に合う方法を選べるようにするために使う。")
    if has_any("検査", "診断", "合理的配慮"):
        important.append("- 検査や診断はラベルを貼るためではなく、学習しやすくする方法や合理的配慮を考えるために使う。")
    for item in important_source:
        if len(important) >= 5:
            break
        if item not in important:
            important.append(item)

    terms = [f"- {term}" for term in topic_terms]
    for term in term_source:
        body = _bullet_body(term)
        if len(body) > 24 or re.search(r"[。！？!?]", body):
            continue
        if body and f"- {body}" not in terms:
            terms.append(f"- {body}")
        if len(terms) >= 10:
            break

    raw_checks = [
        line for line in section_lines(text, "要確認箇所")
        + section_lines(text, "わからなかったかもしれないこと")
        + section_lines(text, "次に先生や大人に確認すること")
        if not is_placeholder_item(line) and "[00:00]" not in line
    ]
    for line in section_lines(text, "英語・専門語・固有名詞"):
        body = _bullet_body(line)
        if body and is_suspicious_review_term(body):
            raw_checks.append(f"- 要確認語: {body}")
    checks: list[str] = []
    seen_checks: set[str] = set()
    for line in raw_checks:
        item = compact_review_item(line) if (
            _bullet_body(line).startswith(("要確認語:", "要確認語：", "要確認断片:", "要確認断片："))
            or is_long_noise_fragment(line)
        ) else line
        if item not in seen_checks:
            seen_checks.add(item)
            checks.append(item)
        if len(checks) >= 5:
            break

    def render(items: list[str], placeholder: str) -> str:
        return "\n".join(items) if items else f"- {placeholder}"

    next_checks = build_adaptive_next_checks(text, dynamic_terms)

    unknown_items = checks[:5]
    for term in terms:
        if term not in unknown_items:
            unknown_items.append(term)
        if len(unknown_items) >= 10:
            break

    return "\n".join([
        "## 今日の内容",
        render(today[:5], "内容を十分に抽出できませんでした。"),
        "",
        "## 大事なこと",
        render(important[:5], "大事なことは抽出できませんでした。"),
        "",
        "## わからなかったかもしれないこと",
        render(unknown_items[:10], "聞き取りにくい箇所は目立ちませんでした。"),
        "",
        "## 次に先生や大人に確認すること",
        render(next_checks[:7], "この記録を学習に使う前に、元の文字起こしと照合してください。"),
    ])


def chunk_list(items: list[str], size: int) -> list[list[str]]:
    size = max(1, size)
    return [items[i:i + size] for i in range(0, len(items), size)]


def build_hierarchical_learner_final(
    llm,
    compact_partials: list[str],
    output_dir: Path,
    log_path: Path,
    group_size: int,
    max_tokens_summary: int,
    max_tokens_final: int,
    max_tokens_core_terms: int,
) -> tuple[str, bool, int, bool, dict]:
    """Create simple map-reduce learner summaries for very long recordings."""
    learner_dir = output_dir / "learner_summaries"
    learner_dir.mkdir(exist_ok=True)
    intermediate: list[str] = []
    fallback_count = 0
    compact_source_text = scrub_review_noise_for_llm("\n\n".join(compact_partials))
    dynamic_core_terms = extract_dynamic_core_terms(
        llm,
        compact_source_text,
        max_tokens_core_terms,
        log_path,
        output_dir,
    )

    for idx, group in enumerate(chunk_list(compact_partials, group_size), start=1):
        group_text = scrub_review_noise_for_llm("\n\n".join(group))
        start = time.perf_counter()
        result = clean_learner_summary(call_llm(
            llm,
            LEARNER_SUMMARY_PROMPT.format(transcript=group_text),
            max_tokens_summary,
        ))
        used_fallback = False
        if not has_learner_sections(result):
            used_fallback = True
            fallback_count += 1
            result = build_deterministic_learner_summary(group_text, dynamic_core_terms)
        seconds = time.perf_counter() - start
        append_log(log_path, f"learner_summary_{idx:03d}_seconds={seconds:.3f} chars_in={len(group_text)} chars_out={len(result)} fallback={used_fallback}")
        path = learner_dir / f"learner_summary_{idx:03d}.md"
        path.write_text(result + "\n", encoding="utf-8")
        intermediate.append(f"<!-- learner summary {idx:03d} -->\n{result}")

    intermediate_text = scrub_review_noise_for_llm("\n\n".join(intermediate))
    quality_source_text = scrub_review_noise_for_llm("\n\n".join(compact_partials + intermediate))
    core_terms_block = "\n".join(f"- {term}" for term in dynamic_core_terms)
    final_prompt_summaries = intermediate_text
    if core_terms_block:
        final_prompt_summaries += "\n\n## この記録で落としてはいけない重要語候補\n" + core_terms_block
    (output_dir / "learner_summaries.md").write_text(intermediate_text + "\n", encoding="utf-8")

    start = time.perf_counter()
    final = clean_learner_summary(call_llm(
        llm,
        LEARNER_FINAL_PROMPT.format(summaries=final_prompt_summaries),
        max_tokens_final,
    ))
    hard_fallback = False
    quality_fallback = False
    metadata = learner_default_quality_metadata()
    metadata["dynamic_core_terms"] = dynamic_core_terms
    metadata["dynamic_core_terms_count"] = len(dynamic_core_terms)
    format_passed = learner_format_check_passed(final)
    metadata["format_check_passed"] = format_passed
    if not format_passed:
        hard_fallback = True
        fallback_count += 1
        final = build_deterministic_learner_summary(intermediate_text, dynamic_core_terms)
        metadata["final_generation_mode"] = "technical_fallback"
        metadata["content_quality_check_passed"] = False
        metadata["quality_gate_reason"] = "format_check_failed"
        metadata["missing_core_terms"] = []
    else:
        quality = learner_content_quality_check(quality_source_text, final, dynamic_core_terms)
        context_check = quality.get("context_check") or detect_learning_record_context(quality_source_text)
        metadata["target_context_check_passed"] = context_check["passed"]
        metadata["target_context_reason"] = context_check["reason"]
        metadata["target_context_positive_hits"] = context_check["positive_hits"]
        metadata["target_context_outside_hits"] = context_check["outside_hits"]
        metadata["content_quality_check_passed"] = quality["passed"]
        metadata["quality_gate_reason"] = quality["reason"]
        metadata["missing_core_terms"] = quality["missing_core_terms"]
        metadata["final_generation_mode"] = "llm"
    if format_passed and not metadata["content_quality_check_passed"]:
        quality_fallback = True
        fallback_count += 1
        final = build_deterministic_learner_summary(quality_source_text, dynamic_core_terms)
        metadata["final_generation_mode"] = "deterministic_merge"
        # Technical format succeeded; the deterministic merge is a learner-quality route.
        metadata["format_check_passed"] = True
    final, checks_completed, checks_count = complete_learner_next_checks(
        final,
        quality_source_text,
        dynamic_core_terms,
    )
    metadata["next_checks_completed"] = checks_completed
    metadata["next_checks_count"] = checks_count
    seconds = time.perf_counter() - start
    append_log(
        log_path,
        " ".join([
            f"learner_final_seconds={seconds:.3f}",
            f"chars_in={len(intermediate_text)}",
            f"chars_out={len(final)}",
            f"hard_fallback={hard_fallback}",
            f"quality_fallback={quality_fallback}",
            f"quality_gate_reason={metadata['quality_gate_reason']}",
            f"missing_core_terms={','.join(metadata['missing_core_terms'])}",
            f"final_generation_mode={metadata['final_generation_mode']}",
            f"target_context_check_passed={metadata['target_context_check_passed']}",
            f"target_context_reason={metadata['target_context_reason']}",
            f"next_checks_completed={checks_completed}",
            f"next_checks_count={checks_count}",
        ]),
    )
    return final, hard_fallback, fallback_count, quality_fallback, metadata


def build_3line_summary_from_record(final_record: str, compact_partial_text: str) -> str:
    """Build ## 3行サマリー from 整形記録 content (prose-based, not term lists)."""
    seikei = [l for l in section_lines(final_record, "整形記録") if not is_placeholder_item(l)]
    if not seikei:
        seikei = [l for l in content_section_lines(compact_partial_text) if not is_placeholder_item(l)][:10]
    checks = [
        l for l in section_lines(final_record, "要確認箇所")
        if "[00:00]" not in l and not is_placeholder_item(l)
    ]

    cleaned = prioritize_summary_items([l.removeprefix("- ").strip() for l in seikei if l.strip().startswith("- ")])
    if len(cleaned) >= 2:
        summary_1 = cleaned[0][:80] + ("…" if len(cleaned[0]) > 80 else "")
        topics: list[str] = []
        for item in cleaned[1:]:
            # Split on clause delimiters to extract short, clean topic phrases
            t = re.split(r"[。、．,]|について|における|のため|により", item)[0].strip()
            if t and len(t) > 3 and t not in _SUMMARY_STOPWORDS:
                topics.append(t[:20])  # 20 chars keeps natural phrase boundaries
            if len(topics) >= 3:
                break
        summary_2 = "主な話題: " + "、".join(topics) if topics else cleaned[1][:80]
    elif len(cleaned) == 1:
        summary_1 = cleaned[0][:80]
        summary_2 = "part_summaries.mdで詳細を確認できます。"
    else:
        summary_1 = "音声記録を整形しました。"
        summary_2 = "part_summaries.mdで詳細を確認できます。"

    # Use only the final record's 要確認箇所 — avoid misleading counts from compact summaries
    summary_3 = f"要確認箇所は{len(checks)}件あります。" if checks else "目立つ要確認箇所はありません。"
    return "\n".join([
        "## 3行サマリー",
        f"- {summary_1}",
        f"- {summary_2}",
        f"- {summary_3}",
    ])


def prioritize_summary_items(items: list[str]) -> list[str]:
    """Prefer learner reasons/experiences over procedural setup in summaries."""
    procedural_markers = ("全体的な流れ", "面接感", "1問", "一問", "職員が1人ずつ", "発表予定")
    personal_markers = PERSONAL_STATEMENT_KEYWORDS
    personal = [item for item in items if any(marker in item for marker in personal_markers)]
    other = [
        item for item in items
        if item not in personal and not any(marker in item for marker in procedural_markers)
    ]
    procedural = [item for item in items if item not in personal and item not in other]
    return personal + other + procedural


def ensure_3line_summary(final_record: str, compact_partial_text: str) -> str:
    """Ensure ## 3行サマリー exists and is content-based prose (not a term list)."""
    if "## 3行サマリー" in final_record:
        items = [l for l in section_lines(final_record, "3行サマリー") if not is_placeholder_item(l)]
        if items and not _is_term_list_summary(final_record):
            return final_record
        final_record = remove_section(final_record, "3行サマリー")
    fallback = build_3line_summary_from_record(final_record, compact_partial_text)
    return final_record.rstrip() + "\n\n" + fallback


def _replace_or_add_section(text: str, heading: str, new_items: list[str]) -> str:
    """Replace an existing empty/missing section with new_items, or append it."""
    lines = text.split("\n")
    in_target = False
    result: list[str] = []
    heading_found = False
    for line in lines:
        is_h2 = bool(re.match(r"^##\s+", line))
        is_target = bool(re.match(rf"^##\s+{re.escape(heading)}\s*$", line))
        if is_target:
            in_target = True
            heading_found = True
            result.append(line)
            result.extend(new_items)
            continue
        if is_h2 and in_target:
            in_target = False
        if not in_target:
            result.append(line)
    if not heading_found:
        result.extend(["", f"## {heading}"] + new_items)
    return "\n".join(result)


def fill_missing_sections_from_parts(final_record: str, compact_partial_text: str) -> str:
    """If FINAL is missing 英語・専門語・固有名詞 or 要確認箇所, inherit from part summaries."""
    def needs_fill(text: str, heading: str) -> bool:
        items = [l for l in section_lines(text, heading) if not is_placeholder_item(l)]
        return len(items) == 0

    result = final_record

    if needs_fill(result, "英語・専門語・固有名詞"):
        terms = [l for l in section_lines(compact_partial_text, "英語・専門語・固有名詞")
                 if not is_placeholder_item(l)]
        seen: set[str] = set()
        unique_terms: list[str] = []
        for t in terms:
            key = t.strip().removeprefix("- ").strip()
            if key not in seen and key not in _TERMS_SECTION_NOISE:
                seen.add(key)
                unique_terms.append(f"- {key}")
        unique_terms = unique_terms[:15]
        if unique_terms:
            result = _replace_or_add_section(result, "英語・専門語・固有名詞", unique_terms)

    if needs_fill(result, "要確認箇所"):
        checks = [l for l in section_lines(compact_partial_text, "要確認箇所")
                  if not is_placeholder_item(l) and "[00:00]" not in l][:10]
        if checks:
            result = _replace_or_add_section(result, "要確認箇所", checks)
        else:
            result = _replace_or_add_section(result, "要確認箇所", ["- 目立つ要確認箇所は抽出されませんでした。"])

    return result


def _clip_line(line: str, max_chars: int) -> str:
    body = line.removeprefix("- ").strip()
    if len(body) > max_chars:
        body = body[:max_chars].rstrip("。、 ") + "…"
        return f"- {body}"
    return line


def build_part_summary(part_record: str) -> str:
    """Create a deterministic compact summary from a formatted part."""
    content_candidates = [
        _clip_line(line, 80)
        for line in section_lines(part_record, "整形記録")
        if not is_placeholder_item(line)
    ]
    terms_raw = [line for line in section_lines(part_record, "英語・専門語・固有名詞") if not is_placeholder_item(line)]
    checks = [
        line for line in section_lines(part_record, "要確認箇所")
        if "[00:00]" not in line and not is_placeholder_item(line)
    ]

    content_candidates, content_checks = split_review_candidates(content_candidates)
    terms: list[str] = []
    term_checks: list[str] = []
    for term in terms_raw:
        body = term.removeprefix("- ").strip()
        if _is_whisper_hallucination_term(body):
            continue
        if is_suspicious_review_term(body):
            term_checks.append(f"- 要確認語: {body}")
            continue
        terms.append(term)
    checks = content_checks + term_checks + checks

    # If the part had little structured content, preserve only topic-bearing terms
    # as content hints. Noisy terms stay in 要確認箇所.
    if len(content_candidates) < 2:
        for term in terms:
            body = term.removeprefix("- ").strip()
            if not has_learner_support_keyword(body):
                continue
            if term not in content_candidates:
                content_candidates.append(term)
            if len(content_candidates) >= 5:
                break

    def render(items: list[str], placeholder: str) -> str:
        return "\n".join(items) if items else f"- {placeholder}"

    return "\n".join([
        f"## {PART_CONTENT_HEADING}",
        render(content_candidates[:8], "この部分の内容は抽出できませんでした。"),
        "",
        "## 英語・専門語・固有名詞",
        render(terms[:8], "該当項目は抽出されませんでした。"),
        "",
        "## 要確認箇所",
        render(checks[:5], "目立つ要確認箇所は抽出されませんでした。"),
    ])


def extract_candidate_terms(text: str) -> list[str]:
    known_terms = [
        "LD",
        "ICT",
        "ADHD",
        "SAT",
        "OCR",
        "Windows",
        "MacBook",
        "iPad",
        "AI",
        "Whisper",
        "通級",
        "通級指導教室",
        "合理的配慮",
        "ディスレクシア",
        "ディスレクチュア",
        "セレクチュア",
        "読み書き障害",
        "学習障害",
        "脳機能",
        "脳の機能",
        "個別の指導計画",
        "言語聴覚士",
        "デジタル教科書",
        "ミライシード",
        "ドリルパーク",
        "テキストスピーチ",
        "ボイスインプット",
        "ルビ",
        "フロリダ",
        "マレーシア",
        "SFC",
        "SOC",
    ]
    found: list[str] = []
    for term in known_terms:
        item = f"- {term}"
        if term in text and item not in found:
            found.append(item)
    katakana = re.findall(r"[ァ-ヴーA-Za-z][ァ-ヴーA-Za-z0-9\-]{3,}", text)
    for term in katakana:
        item = f"- {term}"
        if item not in found:
            found.append(item)
        if len(found) >= 12:
            break
    return found


def block_time_label(block: str) -> str:
    match = re.search(r"^===== (chunk_\d+)(?:\s+([^=]+?))?\s+=====", block, flags=re.MULTILINE)
    if not match:
        return ""
    return (match.group(2) or match.group(1)).strip()


def fallback_part_record(batch: str) -> str:
    blocks = re.split(r"(?=^===== chunk_\d+(?: .+?)? =====$)", batch, flags=re.MULTILINE)
    important: list[str] = []
    checks: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        display = clean_asr_fragment(block_ja_text(block))
        compact_display = compact_text(display)
        if len(compact_display) >= 24:
            important.append(f"- {display[:120]}")
        elif block_has_patch(block):
            patch_label, patch_text = block_patch_text(block)
            if is_filler_fragment(display) and is_filler_fragment(patch_text):
                continue
            label = block_time_label(block)
            detail = display[:100] if display else f"{patch_label or '英語補助認識'}あり。内容確認が必要です。"
            checks.append(f"- [{label}] {detail}")
        if len(important) >= 6:
            break

    terms = extract_candidate_terms(batch)[:10]

    def render(items: list[str], placeholder: str) -> str:
        return "\n".join(items) if items else f"- {placeholder}"

    return "\n".join([
        "## 整形記録",
        render(important[:6], "内容は抽出できませんでした。"),
        "",
        "## 英語・専門語・固有名詞",
        render(terms[:10], "該当項目は抽出されませんでした。"),
        "",
        "## 要確認箇所",
        render(checks[:8], "目立つ要確認箇所は抽出されませんでした。"),
    ])


PERSONAL_STATEMENT_KEYWORDS = (
    "応募した理由",
    "学びたい",
    "勉強したい",
    "伝えたい",
    "生かして",
    "活かして",
    "国際社会",
    "地域活動",
    "カルチャー",
)


def extract_personal_statement_items(batch: str, max_items: int = 4) -> list[str]:
    """Extract learner/candidate reasons, experiences, opinions, and hopes from raw chunks."""
    blocks = re.split(r"(?=^===== chunk_\d+ .+? =====$)", batch, flags=re.MULTILINE)
    candidates: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        ja = re.sub(r"\s+", "", block_ja_text(block))
        if len(ja) < 18:
            continue
        if not any(keyword in ja for keyword in PERSONAL_STATEMENT_KEYWORDS):
            continue
        item = summarize_personal_statement(ja)
        if item not in candidates:
            candidates.append(item)
        if len(candidates) >= max_items:
            break
    return candidates


def summarize_personal_statement(text: str) -> str:
    """Create a compact, source-grounded bullet from a personal statement chunk.

    This only trims the chunk's own text; it does not substitute fixed
    sentences for particular keywords, so the output always reflects what the
    speaker actually said rather than a stock phrase.
    """
    text = text.strip("。、 ")
    return f"- {text[:90]}"


def insert_section_items(text: str, heading: str, items: list[str], max_items: int) -> str:
    if not items:
        return text
    lines = text.split("\n")
    result: list[str] = []
    in_target = False
    inserted = False
    existing_items: list[str] = []
    for line in lines:
        if re.match(r"^##\s+", line):
            if in_target and not inserted:
                for item in items:
                    if item not in existing_items and len(existing_items) < max_items:
                        result.append(item)
                        existing_items.append(item)
                inserted = True
            in_target = bool(re.match(rf"^##\s+{re.escape(heading)}\s*$", line))
            result.append(line)
            continue
        if in_target and line.strip().startswith("- "):
            existing_items.append(line.strip())
        result.append(line)
    if in_target and not inserted:
        for item in items:
            if item not in existing_items and len(existing_items) < max_items:
                result.append(item)
                existing_items.append(item)
    return "\n".join(result)


def rescue_misrouted_sections(text: str) -> str:
    """Move bullets like '- 英語の専門語・固有名詞：...' out of 整形記録 into proper sections.

    Handles LLM quirk where it writes section content as a bullet prefix instead of
    placing items in a separate ## heading section.
    """
    lines = text.split("\n")
    result_lines: list[str] = []
    rescued_terms: list[str] = []
    rescued_checks: list[str] = []
    in_seikei = False

    for line in lines:
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            in_seikei = (heading.group(1) == "整形記録")
            result_lines.append(line)
            continue

        if in_seikei and line.strip().startswith("- "):
            content = line.strip().removeprefix("- ").strip()
            terms_match = re.match(r"(?:英語[の]?専門語[・]固有名詞|英語発話)[：:]\s*(.+)", content)
            checks_match = re.match(r"要確認箇所[：:]\s*(.+)", content)
            if terms_match:
                raw = terms_match.group(1)
                quoted = re.findall(r"「([^」]+)」", raw)
                if quoted:
                    rescued_terms.extend(f"- {t}" for t in quoted)
                else:
                    for t in re.split(r"[、,\s]+", raw):
                        t = t.strip("「」").strip()
                        if t:
                            rescued_terms.append(f"- {t}")
                continue
            elif checks_match:
                raw = checks_match.group(1)
                for item in re.split(r"[、,]", raw):
                    item = item.strip()
                    if item and item not in ("[ ]", "[]"):
                        rescued_checks.append(f"- {item}")
                continue

        result_lines.append(line)

    result = "\n".join(result_lines)
    if rescued_terms:
        result = _replace_or_add_section(result, "英語・専門語・固有名詞", rescued_terms)
    if rescued_checks:
        result = _replace_or_add_section(result, "要確認箇所", rescued_checks)
    return result


def ensure_personal_statements_in_record(text: str, batch: str) -> str:
    """Keep non-formal learner/candidate statements from being dropped by the PART LLM."""
    items = extract_personal_statement_items(batch)
    if not items:
        return text
    compact_record = compact_text(text)
    missing = []
    for item in items:
        compact_item = compact_text(item)
        # Compare by several content words, not the full generated sentence.
        key_hits = sum(1 for keyword in PERSONAL_STATEMENT_KEYWORDS if keyword in item and keyword in compact_record)
        if key_hits == 0 and compact_item[:24] not in compact_record:
            missing.append(item)
    if not missing:
        return text
    return insert_section_items(text, "整形記録", missing, max_items=8)


def split_hybrid_blocks(text: str, max_chars: int) -> list[str]:
    blocks = re.split(r"(?=^===== chunk_\d+(?: .+?)? =====$)", text, flags=re.MULTILINE)
    blocks = [b.strip() for b in blocks if b.strip()]
    if len(blocks) <= 1 and not re.search(r"^===== chunk_\d+(?: .+?)? =====$", text, flags=re.MULTILINE):
        blocks = []
        for unit in re.split(r"\n\s*\n|\n", text):
            unit = unit.strip()
            if not unit:
                continue
            if len(unit) <= max_chars:
                blocks.append(unit)
                continue
            for start in range(0, len(unit), max_chars):
                piece = unit[start : start + max_chars].strip()
                if piece:
                    blocks.append(piece)
    batches: list[str] = []
    current: list[str] = []
    current_len = 0
    for block in blocks:
        extra = len(block) + 2
        if current and current_len + extra > max_chars:
            batches.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(block)
        current_len += extra
    if current:
        batches.append("\n\n".join(current))
    return batches


def compact_text(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"^\s*(ja_base|fw_patch|parakeet_patch|patch_rejected)\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", "", text)
    return text


def patch_labels() -> tuple[str, ...]:
    return ("fw_patch", "parakeet_patch")


def block_has_patch(block: str) -> bool:
    return any(f"[{label}]" in block for label in patch_labels())


def block_ja_text(block: str) -> str:
    match = re.search(r"\[ja_base\](.*?)(?:\[(?:fw_patch|parakeet_patch|patch_rejected)\]|$)", block, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return re.sub(r"^===== chunk_\d+(?: .+?)? =====\s*", "", block.strip(), flags=re.MULTILINE).strip()


def block_label_text(block: str, label: str) -> str:
    pattern = rf"\[{re.escape(label)}\](.*?)(?:\n\[[a-zA-Z_]+\]|\Z)"
    match = re.search(pattern, block, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def block_patch_text(block: str) -> tuple[str, str]:
    for label in patch_labels():
        text = clean_asr_fragment(block_label_text(block, label))
        if text:
            return label, text
    return "", ""


def clean_asr_fragment(text: str) -> str:
    text = text.replace("(empty)", "")
    text = re.sub(r"\[[\d:\.]+\s*-->\s*[\d:\.]+\]\s*", "", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_filler_fragment(text: str) -> bool:
    compact = compact_text(text)
    filler = {
        "あ", "ああ", "あっ", "あれ", "うん", "え", "お", "はい", "いや", "そう",
        "すいません", "よろしく", "ありがとうございます",
    }
    return not compact or compact in filler


def build_formatted_transcript(filtered_text: str) -> tuple[str, str, int]:
    """Build a lightweight ordered transcript without an extra LLM pass."""
    blocks = re.split(r"(?=^===== chunk_\d+(?: .+?)? =====$)", filtered_text, flags=re.MULTILINE)
    md_lines = ["# 整形文字起こし", ""]
    txt_lines: list[str] = []
    written = 0

    for block in [b.strip() for b in blocks if b.strip()]:
        header = re.match(r"^===== (chunk_\d+)(?:\s+([^=]+?))?\s+=====", block)
        if not header:
            continue
        chunk_id = header.group(1)
        time_range = (header.group(2) or "").strip()
        lang_hint = clean_asr_fragment(block_label_text(block, "lang_hint")).lower()
        ja_text = clean_asr_fragment(block_label_text(block, "ja_base"))
        patch_label, patch_text = block_patch_text(block)
        if not ja_text and not patch_text:
            ja_text = clean_asr_fragment(block_ja_text(block))

        chosen = ja_text
        note = ""
        if patch_text and (is_filler_fragment(ja_text) or lang_hint == "en"):
            chosen = patch_text
            note = patch_label
        elif patch_text and ja_text and patch_text not in ja_text and not is_filler_fragment(patch_text):
            chosen = f"{ja_text}\n補助認識: {patch_text}"
            note = f"ja_base + {patch_label}"

        if is_filler_fragment(chosen):
            continue

        heading = f"## {chunk_id} {time_range}".rstrip()
        md_lines.extend([heading, chosen])
        if note:
            md_lines.append(f"認識: {note}")
        md_lines.append("")
        prefix = f"[{time_range}] " if time_range else f"[{chunk_id}] "
        txt_lines.append(f"{prefix}{chosen}")
        written += 1

    if written == 0:
        plain_lines = [
            clean_asr_fragment(line)
            for line in filtered_text.splitlines()
            if clean_asr_fragment(line)
        ]
        for line in plain_lines:
            md_lines.append(line)
            txt_lines.append(line)
        written = len(plain_lines)

    return "\n".join(md_lines).rstrip() + "\n", "\n\n".join(txt_lines).rstrip() + "\n", written


def filter_school_relevant_blocks(text: str) -> str:
    """Remove obvious filler-only chunks before sending text to the LLM."""
    filler = {
        "あ",
        "ああ",
        "あっ",
        "あれ",
        "うん",
        "え",
        "お",
        "はい",
        "いや",
        "そう",
        "フフ",
    }
    blocks = re.split(r"(?=^===== chunk_\d+(?: .+?)? =====$)", text, flags=re.MULTILINE)
    kept: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if block_has_patch(block):
            ja_fragment = clean_asr_fragment(block_ja_text(block))
            _patch_label, patch_fragment = block_patch_text(block)
            if not is_filler_fragment(patch_fragment) or not is_filler_fragment(ja_fragment):
                kept.append(block)
            continue
        ja = compact_text(block_ja_text(block))
        if len(ja) >= 12 and ja not in filler:
            kept.append(block)
    return "\n\n".join(kept)


_ASSISTANT_PREFIX = "## 整形記録\n"
_CORE_TERMS_ASSISTANT_PREFIX = "- "
_CLEAN_ASSISTANT_PREFIX = "# クリーン文字起こし\n"

# Token IDs for <think>/<think> suppression via logit_bias.
# Qwen3 (1.7B, 0.6B): 151667 = <think>, 151668 = </think>
# Qwen3.5 (2B):       248068 = <think>, 248069 = </think>
# Including both sets so the same dict works for all supported models.
_NO_THINK_LOGIT_BIAS = {
    151667: -100.0,  # Qwen3
    151668: -100.0,  # Qwen3
    248068: -100.0,  # Qwen3.5
    248069: -100.0,  # Qwen3.5
}
_SAMPLING_TEMPERATURE = 0.0
_SAMPLING_TOP_P = 0.95
_SAMPLING_TOP_K = 40


def call_llm(llm, user_prompt: str, max_tokens: int) -> str:
    vocab_size = llm.n_vocab()
    logit_bias = {k: v for k, v in _NO_THINK_LOGIT_BIAS.items() if k < vocab_size}
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": _ASSISTANT_PREFIX},
        ],
        temperature=_SAMPLING_TEMPERATURE,
        top_p=_SAMPLING_TOP_P,
        top_k=_SAMPLING_TOP_K,
        repeat_penalty=1.15,
        max_tokens=max_tokens,
        logit_bias=logit_bias,
    )
    return clean_formatted_output(clean_llm_text(response["choices"][0]["message"]["content"]))


def call_llm_core_terms(llm, user_prompt: str, max_tokens: int) -> str:
    vocab_size = llm.n_vocab()
    logit_bias = {k: v for k, v in _NO_THINK_LOGIT_BIAS.items() if k < vocab_size}
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": _CORE_TERMS_ASSISTANT_PREFIX},
        ],
        temperature=_SAMPLING_TEMPERATURE,
        top_p=_SAMPLING_TOP_P,
        top_k=_SAMPLING_TOP_K,
        repeat_penalty=1.12,
        max_tokens=max_tokens,
        logit_bias=logit_bias,
    )
    text = clean_llm_text(response["choices"][0]["message"]["content"])
    if text and not text.lstrip().startswith("- "):
        text = "- " + text.lstrip()
    return text.strip()


def call_llm_clean(llm, user_prompt: str, max_tokens: int) -> str:
    vocab_size = llm.n_vocab()
    logit_bias = {k: v for k, v in _NO_THINK_LOGIT_BIAS.items() if k < vocab_size}
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": _CLEAN_ASSISTANT_PREFIX},
        ],
        temperature=_SAMPLING_TEMPERATURE,
        top_p=_SAMPLING_TOP_P,
        top_k=_SAMPLING_TOP_K,
        repeat_penalty=1.12,
        max_tokens=max_tokens,
        logit_bias=logit_bias,
    )
    text = clean_llm_text(response["choices"][0]["message"]["content"])
    if not text.startswith("# クリーン文字起こし"):
        text = "# クリーン文字起こし\n" + text.lstrip()
    return text.strip()


def main() -> int:
    global _SAMPLING_TEMPERATURE, _SAMPLING_TOP_P, _SAMPLING_TOP_K
    args = parse_args()
    _SAMPLING_TEMPERATURE = args.temperature
    _SAMPLING_TOP_P = args.top_p
    _SAMPLING_TOP_K = args.top_k
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    raw = Path(args.input).read_text(encoding="utf-8", errors="replace")
    raw_transcript_path = output_dir / "raw_transcript.txt"
    raw_transcript_path.write_text(raw.rstrip() + "\n", encoding="utf-8")
    filtered = filter_school_relevant_blocks(raw)
    (output_dir / "filtered_input.txt").write_text(filtered + "\n", encoding="utf-8")
    formatted_md, formatted_txt, formatted_chunks = build_formatted_transcript(filtered)
    formatted_md_path = output_dir / "formatted_transcript.md"
    formatted_txt_path = output_dir / "formatted_transcript.txt"
    formatted_md_path.write_text(formatted_md, encoding="utf-8")
    formatted_txt_path.write_text(formatted_txt, encoding="utf-8")
    batches = split_hybrid_blocks(filtered, args.max_chars_per_batch)
    if args.max_batches > 0:
        batches = batches[: args.max_batches]

    prompt_reference = ""
    if args.prompt_file:
        prompt_reference = Path(args.prompt_file).read_text(encoding="utf-8", errors="replace")
        (output_dir / "prompt_used.md").write_text(prompt_reference, encoding="utf-8")

    append_log(log_path, "school_hybrid_postprocess_start")
    append_log(log_path, f"input={args.input}")
    append_log(log_path, f"model={args.model}")
    append_log(log_path, "thinking_mode=off")
    append_log(log_path, f"temperature={args.temperature}")
    append_log(log_path, f"top_p={args.top_p}")
    append_log(log_path, f"top_k={args.top_k}")
    append_log(log_path, f"clean_mode={args.clean_mode}")
    append_log(log_path, f"batches={len(batches)}")
    append_log(log_path, f"max_chars_per_batch={args.max_chars_per_batch}")

    if len(batches) == 0:
        append_log(log_path, "no_substantive_batches=true")
        (output_dir / "chunks_raw").mkdir(exist_ok=True)
        (output_dir / "chunks_safe").mkdir(exist_ok=True)
        (output_dir / "chunks_clean").mkdir(exist_ok=True)
        safe_transcript_path = output_dir / "safe_transcript.md"
        safe_transcript_path.write_text(ensure_safe_transcript_note(formatted_md), encoding="utf-8")
        clean_transcript_path = output_dir / "clean_transcript.md"
        clean_transcript_path.write_text(ensure_safe_transcript_note(formatted_md), encoding="utf-8")
        final_record = "\n".join([
            "## 整形記録",
            "- 有効な発話は抽出されませんでした。",
            "",
            "## 英語・専門語・固有名詞",
            "- 該当項目は抽出できませんでした。",
            "",
            "## 要確認箇所",
            "- 目立つ要確認箇所は抽出されませんでした。",
        ])
        final_record = ensure_3line_summary(final_record, "")
        final_record = ensure_meeting_record_note(final_record)
        final_path = output_dir / "school_record.md"
        final_path.write_text(final_record + "\n", encoding="utf-8")
        meeting_record_path = output_dir / "meeting_record.md"
        meeting_record_path.write_text(final_record + "\n", encoding="utf-8")
        review_flags_path = output_dir / "review_flags.md"
        review_flags_path.write_text(build_review_flags("", final_record), encoding="utf-8")
        summary_md = render_section_only(final_record, "3行サマリー") + "\n"
        summary_txt = "\n".join(
            item.removeprefix("- ").strip()
            for item in section_lines(final_record, "3行サマリー")
        ).rstrip() + "\n"
        summary_md_path = output_dir / "summary.md"
        summary_txt_path = output_dir / "summary.txt"
        summary_md_path.write_text(summary_md, encoding="utf-8")
        summary_txt_path.write_text(summary_txt, encoding="utf-8")
        (output_dir / "partial_records.md").write_text("", encoding="utf-8")
        (output_dir / "part_summaries.md").write_text("", encoding="utf-8")
        result_summary = {
            "input": args.input,
            "model": args.model,
            "thinking_mode": "off",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "batches": 0,
            "formatted_chunks": formatted_chunks,
            "n_ctx": args.n_ctx,
            "max_chars_per_batch": args.max_chars_per_batch,
            "model_load_seconds": 0,
            "part_summary_seconds": 0,
            "final_seconds": 0,
            "final_used_fallback": True,
            "raw_transcript": str(raw_transcript_path),
            "safe_transcript": str(safe_transcript_path),
            "ai_readable_transcript": "",
            "primary_transcript": str(safe_transcript_path),
            "clean_transcript": str(clean_transcript_path),
            "chunks_raw": str(output_dir / "chunks_raw"),
            "chunks_safe": str(output_dir / "chunks_safe"),
            "chunks_ai_readable": "",
            "chunks_clean": str(output_dir / "chunks_clean"),
            "school_record": str(final_path),
            "meeting_record": str(meeting_record_path),
            "review_flags": str(review_flags_path),
            "formatted_transcript_md": str(formatted_md_path),
            "formatted_transcript_txt": str(formatted_txt_path),
            "summary_md": str(summary_md_path),
            "summary_txt": str(summary_txt_path),
            "partial_records": str(output_dir / "partial_records.md"),
            "part_summaries": str(output_dir / "part_summaries.md"),
        }
        (output_dir / "school_postprocess_summary.json").write_text(json.dumps(result_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        append_log(log_path, "school_hybrid_postprocess_done_no_batches")
        print(json.dumps(result_summary, ensure_ascii=False, indent=2), flush=True)
        return 0

    llm = None
    load_sec = 0.0
    if args.clean_mode == "llm" or not args.stage1_only:
        from llama_cpp import Llama

        print(f"Loading model: {args.model}", flush=True)
        print_progress("load")
        load_start = time.perf_counter()
        llm = Llama(
            model_path=args.model,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_batch=args.n_batch,
            n_gpu_layers=args.n_gpu_layers,
            verbose=False,
        )
        load_sec = time.perf_counter() - load_start
        append_log(log_path, f"model_load_seconds={load_sec:.3f}")
    else:
        append_log(log_path, "model_load_skipped=true")

    print("Creating safe transcript", flush=True)
    clean_start = time.perf_counter()
    raw_chunk_dir = output_dir / "chunks_raw"
    safe_dir = output_dir / "chunks_safe"
    clean_dir = output_dir / "chunks_clean"  # legacy compatibility
    ai_dir = output_dir / "chunks_ai_readable"
    raw_chunk_dir.mkdir(exist_ok=True)
    safe_dir.mkdir(exist_ok=True)
    clean_dir.mkdir(exist_ok=True)
    if args.clean_mode == "llm":
        ai_dir.mkdir(exist_ok=True)
    clean_batches: list[str] = []
    safe_batches: list[str] = []
    ai_batches: list[str] = []
    for idx, batch in enumerate(batches, start=1):
        print(f"Cleaning batch {idx}/{len(batches)}", flush=True)
        print_progress("clean", current=idx, total=len(batches))
        start = time.perf_counter()
        raw_chunk_path = raw_chunk_dir / f"chunk_{idx:03d}_raw.txt"
        raw_chunk_path.write_text(batch.rstrip() + "\n", encoding="utf-8")
        chunk_time = f"part_{idx:03d}"
        safe_result = deterministic_clean_transcript(batch)
        safe_batches.append(safe_result)
        safe_path = safe_dir / f"chunk_{idx:03d}_safe.md"
        safe_path.write_text(safe_result.rstrip() + "\n", encoding="utf-8")
        clean_fallback = args.clean_mode == "safe"
        if args.clean_mode == "llm" and llm is not None:
            prompt = CLEAN_USER_PROMPT.format(chunk_time=chunk_time, transcript=batch)
            clean_result = sanitize_clean_transcript(call_llm_clean(llm, prompt, args.max_tokens_clean))
        else:
            clean_result = safe_result
        if args.clean_mode == "llm" and (len(compact_text(clean_result)) < 12 or clean_transcript_has_inference(clean_result)):
            clean_result = safe_result
            clean_fallback = True
        seconds = time.perf_counter() - start
        append_log(log_path, f"clean_{idx:03d}_seconds={seconds:.3f} chars_in={len(batch)} chars_out={len(clean_result)} fallback={clean_fallback}")
        clean_path = clean_dir / f"chunk_{idx:03d}_clean.md"
        clean_path.write_text(clean_result.rstrip() + "\n", encoding="utf-8")
        clean_batches.append(clean_result)
        if args.clean_mode == "llm":
            ai_path = ai_dir / f"chunk_{idx:03d}_ai_readable.md"
            ai_path.write_text(clean_result.rstrip() + "\n", encoding="utf-8")
            ai_batches.append(clean_result)

    safe_transcript_text = ensure_safe_transcript_note("\n\n".join(safe_batches))
    safe_transcript_path = output_dir / "safe_transcript.md"
    safe_transcript_path.write_text(safe_transcript_text, encoding="utf-8")
    if args.clean_mode == "llm":
        clean_transcript_text = ensure_ai_readable_transcript_note("\n\n".join(ai_batches))
        ai_readable_transcript_path = output_dir / "ai_readable_transcript.md"
        ai_readable_transcript_path.write_text(clean_transcript_text, encoding="utf-8")
        primary_transcript_path = ai_readable_transcript_path
    else:
        clean_transcript_text = safe_transcript_text
        ai_readable_transcript_path = None
        primary_transcript_path = safe_transcript_path
    clean_transcript_path = output_dir / "clean_transcript.md"
    clean_transcript_path.write_text(clean_transcript_text, encoding="utf-8")
    clean_sec = time.perf_counter() - clean_start
    append_log(log_path, f"clean_total_seconds={clean_sec:.3f}")

    if args.stage1_only:
        review_flags_path = output_dir / "review_flags.md"
        review_flags_path.write_text(build_review_flags(safe_transcript_text, ""), encoding="utf-8")
        stage1_summary = {
            "input": args.input,
            "model": args.model,
            "thinking_mode": "off",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "stage": "stage1_safe_transcript",
            "stage1_only": True,
            "clean_mode": args.clean_mode,
            "batches": len(batches),
            "formatted_chunks": formatted_chunks,
            "n_ctx": args.n_ctx,
            "max_chars_per_batch": args.max_chars_per_batch,
            "model_load_seconds": round(load_sec, 3),
            "clean_seconds": round(clean_sec, 3),
            "raw_transcript": str(raw_transcript_path),
            "safe_transcript": str(safe_transcript_path),
            "ai_readable_transcript": str(ai_readable_transcript_path) if ai_readable_transcript_path else "",
            "primary_transcript": str(primary_transcript_path),
            "clean_transcript": str(clean_transcript_path),
            "review_flags": str(review_flags_path),
            "chunks_raw": str(output_dir / "chunks_raw"),
            "chunks_safe": str(output_dir / "chunks_safe"),
            "chunks_ai_readable": str(output_dir / "chunks_ai_readable") if args.clean_mode == "llm" else "",
            "chunks_clean": str(output_dir / "chunks_clean"),
            "formatted_transcript_md": str(formatted_md_path),
            "formatted_transcript_txt": str(formatted_txt_path),
            "local_only": True,
            "needs_human_review": True,
        }
        (output_dir / "stage1_summary.json").write_text(json.dumps(stage1_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "school_postprocess_summary.json").write_text(json.dumps(stage1_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        append_log(log_path, "school_hybrid_postprocess_done_stage1_only")
        print(json.dumps(stage1_summary, ensure_ascii=False, indent=2), flush=True)
        return 0

    raw_parts: list[str] = []
    partials: list[str] = []
    part_dir = output_dir / "parts"
    part_dir.mkdir(exist_ok=True)
    for idx, batch in enumerate(batches, start=1):
        print(f"Formatting batch {idx}/{len(batches)}", flush=True)
        print_progress("part", current=idx, total=len(batches))
        start = time.perf_counter()
        clean_batch = clean_batches[idx - 1] if idx - 1 < len(clean_batches) else batch
        prompt = PART_USER_PROMPT.format(transcript=clean_batch)
        try:
            result = call_llm(llm, prompt, args.max_tokens_part)
            result = rescue_misrouted_sections(result)
            result = absorb_preamble_as_record_items(result)
            result = sanitize_unsupported_roles(result, batch)
            result = sanitize_unsupported_context_labels(result, batch)
            result = normalize_known_terms(result)
            result = ensure_personal_statements_in_record(result, batch)
            result = ensure_temporal_coverage_in_record(result, clean_batch, max_items=8)
            result = limit_section_items(result, {"整形記録": 8, "英語・専門語・固有名詞": 10, "要確認箇所": 8})
            result = strip_arrow_quote_format(result)
            result = split_commajoined_terms(result)
            result = clip_record_item_length(result)
            result = fill_missing_part_english_terms(result, batch)
        except Exception as exc:
            append_log(log_path, f"batch_{idx:03d}_technical_fallback={type(exc).__name__}: {exc}")
            result = fallback_part_record(batch)
            result = sanitize_unsupported_roles(result, batch)
            result = sanitize_unsupported_context_labels(result, batch)
            result = normalize_known_terms(result)
            result = ensure_personal_statements_in_record(result, batch)
            result = clip_record_item_length(result)
            result = fill_missing_part_english_terms(result, batch)
        if not has_substantive_section(result, "整形記録"):
            append_log(log_path, f"batch_{idx:03d}_fallback_used=true")
            result = fallback_part_record(batch)
            result = sanitize_unsupported_roles(result, batch)
            result = sanitize_unsupported_context_labels(result, batch)
            result = normalize_known_terms(result)
            result = ensure_personal_statements_in_record(result, batch)
            result = clip_record_item_length(result)
            result = fill_missing_part_english_terms(result, batch)
        seconds = time.perf_counter() - start
        append_log(log_path, f"batch_{idx:03d}_seconds={seconds:.3f} chars_in={len(batch)} chars_out={len(result)}")
        part_path = part_dir / f"part_{idx:03d}.md"
        part_path.write_text(result + "\n", encoding="utf-8")
        raw_parts.append(result)
        partials.append(f"<!-- part {idx:03d} -->\n{result}")

    partial_text = "\n\n".join(partials)
    (output_dir / "partial_records.md").write_text(partial_text + "\n", encoding="utf-8")

    print("Creating compact part summaries", flush=True)
    summary_start = time.perf_counter()
    compact_partials: list[str] = []
    summary_dir = output_dir / "part_summaries"
    summary_dir.mkdir(exist_ok=True)
    for idx, part_record in enumerate(partials, start=1):
        print(f"Compacting part {idx}/{len(partials)}", flush=True)
        print_progress("compact", current=idx, total=len(partials))
        start = time.perf_counter()
        summary_result = build_part_summary(part_record)
        summary_result = sanitize_unsupported_roles(summary_result, part_record)
        summary_result = sanitize_unsupported_context_labels(summary_result, part_record)
        summary_result = normalize_known_terms(summary_result)
        summary_result = limit_section_items(summary_result, {PART_CONTENT_HEADING: 8, "英語・専門語・固有名詞": 8, "要確認箇所": 5})
        seconds = time.perf_counter() - start
        append_log(log_path, f"summary_{idx:03d}_seconds={seconds:.3f} chars_in={len(part_record)} chars_out={len(summary_result)}")
        summary_path = summary_dir / f"summary_{idx:03d}.md"
        summary_path.write_text(summary_result + "\n", encoding="utf-8")
        compact_partials.append(f"<!-- part summary {idx:03d} -->\n{summary_result}")

    compact_partial_text = "\n\n".join(compact_partials)
    (output_dir / "part_summaries.md").write_text(compact_partial_text + "\n", encoding="utf-8")
    summary_sec = time.perf_counter() - summary_start
    append_log(log_path, f"part_summary_total_seconds={summary_sec:.3f}")

    print("Creating final integrated school record", flush=True)
    print_progress("merge")
    final_start = time.perf_counter()
    final_used_fallback = False
    use_learner_final = args.learner_final_threshold > 0 and len(raw_parts) >= args.learner_final_threshold
    final_strategy = "hierarchical_learner_final" if use_learner_final else ("single_part" if len(raw_parts) == 1 else "classic_final")
    learner_final_used = False
    learner_final_fallback_count = 0
    learner_final_quality_fallback = False
    learner_quality_metadata = learner_default_quality_metadata()
    if use_learner_final:
        learner_final_used = True
        (
            final_record,
            learner_final_hard_fallback,
            learner_final_fallback_count,
            learner_final_quality_fallback,
            learner_quality_metadata,
        ) = build_hierarchical_learner_final(
            llm,
            compact_partials,
            output_dir,
            log_path,
            args.learner_final_group_size,
            args.max_tokens_learner_summary,
            args.max_tokens_learner_final,
            args.max_tokens_core_terms,
        )
        final_used_fallback = learner_final_hard_fallback
    elif len(raw_parts) == 1:
        # Single batch: use part record directly, no FINAL LLM needed.
        final_record = raw_parts[0]
        final_record = limit_section_items(final_record, {"整形記録": 10, "英語・専門語・固有名詞": 15, "要確認箇所": 10})
        if not has_substantive_section(final_record, "整形記録"):
            append_log(log_path, "final_fallback_used=true")
            final_record = fallback_final_record(compact_partial_text)
            final_record = sanitize_unsupported_roles(final_record, raw)
            final_record = sanitize_unsupported_context_labels(final_record, raw)
            final_record = normalize_known_terms(final_record)
            final_used_fallback = True
    else:
        # Multiple batches: use FINAL LLM to integrate part summaries.
        final_input = compact_partial_text
        if len(final_input) > 18000:
            final_input = final_input[:18000] + "\n\n[注: 分割整形結果が長いため、ここ以降は個別partを参照してください。]"
        final_record = strip_reasoning_preamble(call_llm(
            llm,
            FINAL_USER_PROMPT.format(partial_records=final_input),
            args.max_tokens_final,
        ))
        final_record = sanitize_unsupported_roles(final_record, raw)
        final_record = sanitize_unsupported_context_labels(final_record, raw)
        final_record = normalize_known_terms(final_record)
        final_record = limit_section_items(final_record, {"整形記録": 10, "英語・専門語・固有名詞": 15, "要確認箇所": 10})
        final_record = strip_arrow_quote_format(final_record)
        final_record = split_commajoined_terms(final_record)
        final_record = clip_record_item_length(final_record)
        if not has_substantive_section(final_record, "整形記録"):
            append_log(log_path, "final_fallback_used=true")
            final_record = fallback_final_record(compact_partial_text)
            final_record = sanitize_unsupported_roles(final_record, raw)
            final_record = sanitize_unsupported_context_labels(final_record, raw)
            final_record = normalize_known_terms(final_record)
            final_record = clip_record_item_length(final_record)
            final_used_fallback = True
    if has_learner_sections(final_record):
        final_record = clean_learner_summary(final_record)
    else:
        final_record = fill_missing_sections_from_parts(final_record, compact_partial_text)
        final_record = ensure_3line_summary(final_record, compact_partial_text)
    final_record = normalize_known_terms(final_record)
    final_record = ensure_meeting_record_note(final_record)
    if final_used_fallback and not has_substantive_section(final_record, "整形記録"):
        final_record = mark_record_for_reprocess(
            final_record,
            "FINAL統合が十分な整形記録を返さなかったため、決定的fallbackで作成しました。",
        )
    final_sec = time.perf_counter() - final_start
    append_log(log_path, f"final_seconds={final_sec:.3f} chars_out={len(final_record)}")

    final_path = output_dir / "school_record.md"
    final_path.write_text(final_record + "\n", encoding="utf-8")
    meeting_record_path = output_dir / "meeting_record.md"
    meeting_record_path.write_text(final_record + "\n", encoding="utf-8")
    review_flags_path = output_dir / "review_flags.md"
    review_flags_path.write_text(build_review_flags(safe_transcript_text, final_record), encoding="utf-8")
    learner_quality_gate_path = output_dir / "learner_quality_gate.md"
    learner_quality_gate_path.write_text(build_learner_quality_gate_report(learner_quality_metadata), encoding="utf-8")
    summary_md = render_section_only(final_record, "3行サマリー") + "\n"
    summary_txt = "\n".join(
        item.removeprefix("- ").strip()
        for item in section_lines(final_record, "3行サマリー")
    ).rstrip() + "\n"
    summary_md_path = output_dir / "summary.md"
    summary_txt_path = output_dir / "summary.txt"
    summary_md_path.write_text(summary_md, encoding="utf-8")
    summary_txt_path.write_text(summary_txt, encoding="utf-8")

    summary = {
        "input": args.input,
        "model": args.model,
        "thinking_mode": "off",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "clean_mode": args.clean_mode,
        "batches": len(batches),
        "formatted_chunks": formatted_chunks,
        "n_ctx": args.n_ctx,
        "max_chars_per_batch": args.max_chars_per_batch,
        "model_load_seconds": round(load_sec, 3),
        "clean_seconds": round(clean_sec, 3),
        "part_summary_seconds": round(summary_sec, 3),
        "final_seconds": round(final_sec, 3),
        "final_strategy": final_strategy,
        "learner_final_used": learner_final_used,
        "learner_final_fallback_count": learner_final_fallback_count,
        "learner_final_quality_fallback": learner_final_quality_fallback,
        "format_check_passed": learner_quality_metadata["format_check_passed"],
        "content_quality_check_passed": learner_quality_metadata["content_quality_check_passed"],
        "quality_gate_reason": learner_quality_metadata["quality_gate_reason"],
        "missing_core_terms": learner_quality_metadata["missing_core_terms"],
        "dynamic_core_terms": learner_quality_metadata["dynamic_core_terms"],
        "dynamic_core_terms_count": learner_quality_metadata["dynamic_core_terms_count"],
        "target_context_check_passed": learner_quality_metadata["target_context_check_passed"],
        "target_context_reason": learner_quality_metadata["target_context_reason"],
        "target_context_positive_hits": learner_quality_metadata["target_context_positive_hits"],
        "target_context_outside_hits": learner_quality_metadata["target_context_outside_hits"],
        "next_checks_completed": learner_quality_metadata["next_checks_completed"],
        "next_checks_count": learner_quality_metadata["next_checks_count"],
        "final_generation_mode": learner_quality_metadata["final_generation_mode"],
        "final_used_fallback": final_used_fallback,
        "raw_transcript": str(raw_transcript_path),
        "safe_transcript": str(safe_transcript_path),
        "ai_readable_transcript": str(ai_readable_transcript_path) if ai_readable_transcript_path else "",
        "primary_transcript": str(primary_transcript_path),
        "clean_transcript": str(clean_transcript_path),
        "chunks_raw": str(output_dir / "chunks_raw"),
        "chunks_safe": str(output_dir / "chunks_safe"),
        "chunks_ai_readable": str(output_dir / "chunks_ai_readable") if args.clean_mode == "llm" else "",
        "chunks_clean": str(output_dir / "chunks_clean"),
        "school_record": str(final_path),
        "meeting_record": str(meeting_record_path),
        "review_flags": str(review_flags_path),
        "learner_quality_gate": str(learner_quality_gate_path),
        "formatted_transcript_md": str(formatted_md_path),
        "formatted_transcript_txt": str(formatted_txt_path),
        "summary_md": str(summary_md_path),
        "summary_txt": str(summary_txt_path),
        "partial_records": str(output_dir / "partial_records.md"),
        "part_summaries": str(output_dir / "part_summaries.md"),
    }
    (output_dir / "school_postprocess_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    append_log(log_path, "school_hybrid_postprocess_done")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
