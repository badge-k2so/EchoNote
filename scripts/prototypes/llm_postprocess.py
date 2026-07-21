"""LLM post-processing: format ASR transcript and generate summary (separate outputs)."""

import argparse
import json
import re
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",              required=True,  help="Input transcript .txt file")
    parser.add_argument("--output_dir",         required=True,  help="Output directory")
    parser.add_argument("--model",              required=True,  help="GGUF model path")
    parser.add_argument("--log",                required=True,  help="Log file path")
    parser.add_argument("--n_ctx",              type=int, default=4096)
    parser.add_argument("--max_tokens_format",  type=int, default=2048)
    parser.add_argument("--max_tokens_summary", type=int, default=512)
    return parser.parse_args()


def append_log(log_path: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{stamp} {message}\n")


NO_THINK = "思考過程、推論メモ、<think>タグは出力しないでください。最終回答だけを日本語で出力してください。"

FORMAT_SYSTEM = f"あなたは音声認識の文字起こし整形の専門家です。{NO_THINK}"
FORMAT_USER = """\
以下は音声認識によって出力された文字起こしテキストです。
次のルールに従って整形してください：
- 句読点（。、）を適切な位置に補う
- 明らかな誤認識や変換ミスを修正する（例：「単にの先生」→「担任の先生」）
- 話し言葉の繰り返しや言い淀みは軽くまとめる
- 内容の追加・削除・解釈の変更はしない
- 発言ごとに段落で区切って読みやすくする

【文字起こし】
{transcript}

【整形後のテキスト】
/no_think"""

SUMMARY_SYSTEM = f"あなたは授業・面談記録の要約の専門家です。{NO_THINK}"
SUMMARY_USER = """\
以下は授業または面談の文字起こしテキストです。
テキスト全体を読んで、話し合いの内容を3〜5項目の箇条書きで簡潔にまとめてください。
各チャンクの繰り返しではなく、全体として何が話されたかを要約してください。
箇条書きは「・」で始めてください。

【文字起こし】
{transcript}

【要約（3〜5項目）】
/no_think"""

_NO_THINK_LOGIT_BIAS = {
    151667: -100.0,  # Qwen3 <think>
    151668: -100.0,  # Qwen3 </think>
    248068: -100.0,  # Qwen3.5 <think>
    248069: -100.0,  # Qwen3.5 </think>
}


def call_llm(llm, system_msg: str, user_msg: str, max_tokens: int) -> str:
    vocab_size = llm.n_vocab()
    logit_bias = {k: v for k, v in _NO_THINK_LOGIT_BIAS.items() if k < vocab_size}
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
        logit_bias=logit_bias,
    )
    text = response["choices"][0]["message"]["content"].strip()
    # Qwen3 thinking mode の <think>...</think> ブロックを除去。
    # 最大トークンに達すると閉じタグが出ないことがあるため、未閉じブロックも落とす。
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    # モデルがプロンプトの見出し（【...】）を冒頭・末尾に返すことがあるため除去
    lines = text.splitlines()
    lines = [l for l in lines if not (l.strip().startswith("【") and l.strip().endswith("】"))]
    return "\n".join(lines).strip()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = Path(args.input).read_text(encoding="utf-8")
    # チャンクヘッダー（===== chunk_000 ===== 等）とタイムスタンプ行（[0.00 --> 0.00]）を除去
    transcript = re.sub(r"^={3,}.*={3,}\s*$", "", raw, flags=re.MULTILINE)
    transcript = re.sub(r"^\[\d+\.\d+ --> \d+\.\d+\]\s*", "", transcript, flags=re.MULTILINE)
    transcript = re.sub(r"\n{3,}", "\n\n", transcript).strip()

    append_log(log_path, "llm_postprocess_start")
    append_log(log_path, f"model={args.model}")
    append_log(log_path, "thinking_mode=off")
    append_log(log_path, f"input={args.input} chars={len(transcript)}")

    from llama_cpp import Llama

    print(f"Loading model: {args.model}", flush=True)
    load_start = time.perf_counter()
    llm = Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_threads=None,
        verbose=False,
    )
    load_sec = time.perf_counter() - load_start
    append_log(log_path, f"model_load_seconds={load_sec:.3f}")
    print(f"Model loaded in {load_sec:.3f}s", flush=True)

    # --- 整形 ---
    print("Running: format (整形) ...", flush=True)
    fmt_start = time.perf_counter()
    formatted = call_llm(llm, FORMAT_SYSTEM,
                         FORMAT_USER.format(transcript=transcript),
                         args.max_tokens_format)
    fmt_sec = time.perf_counter() - fmt_start
    append_log(log_path, f"format_seconds={fmt_sec:.3f}")
    print(f"Format done in {fmt_sec:.3f}s", flush=True)

    fmt_path = output_dir / "formatted.txt"
    fmt_path.write_text(formatted, encoding="utf-8")
    print(f"Formatted -> {fmt_path}", flush=True)

    # --- 要約 ---
    print("Running: summary (要約) ...", flush=True)
    sum_start = time.perf_counter()
    summary = call_llm(llm, SUMMARY_SYSTEM,
                       SUMMARY_USER.format(transcript=transcript),
                       args.max_tokens_summary)
    sum_sec = time.perf_counter() - sum_start
    append_log(log_path, f"summary_seconds={sum_sec:.3f}")
    print(f"Summary done in {sum_sec:.3f}s", flush=True)

    sum_path = output_dir / "summary.txt"
    sum_path.write_text(summary, encoding="utf-8")
    print(f"Summary   -> {sum_path}", flush=True)

    result = {
        "model": args.model,
        "thinking_mode": "off",
        "input_chars": len(transcript),
        "model_load_seconds": round(load_sec, 3),
        "format_seconds":     round(fmt_sec, 3),
        "summary_seconds":    round(sum_sec, 3),
        "formatted": str(fmt_path),
        "summary":   str(sum_path),
    }
    (output_dir / "llm_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    append_log(log_path, f"formatted={fmt_path}")
    append_log(log_path, f"summary={sum_path}")
    append_log(log_path, "llm_postprocess_done")

    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
