from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

from llama_cpp import Llama

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from otoweave_app.asr import select_asr_threads
from otoweave_app.llm_chat import (
    build_initial_messages,
    chat_one_turn,
    find_relevant_transcript_excerpts,
    load_context,
)


CASES = [
    {
        "id": "overview",
        "question": "今日の主な内容を短く教えてください。",
        "required_groups": [["読み書き", "合理的配慮", "ICT", "オンライン"]],
    },
    {
        "id": "diagnosis_purpose",
        "question": "検査や診断は何のために使うと説明されましたか。",
        "required_groups": [["学習", "学び"], ["方法", "合理的配慮"]],
    },
    {
        "id": "ict_support",
        "question": "ICTやオンラインは子どもの何を助けると説明されましたか。",
        "required_groups": [["参加"], ["表現"]],
    },
    {
        "id": "hearing_screening",
        "question": "新生児聴覚スクリーニングについて何と言っていましたか。",
        "required_groups": [["新生児聴覚スクリーニング"], ["健診", "組み込"]],
    },
    {
        "id": "accommodation_request",
        "question": "合理的配慮は誰の申し入れがあった時に提供すると話していましたか。",
        "required_groups": [["本人"], ["申し入れ"]],
    },
    {
        "id": "hospital_stay",
        "question": "日本の精神病院の入院日数について何と言っていましたか。",
        "required_groups": [["入院日数"], ["多", "上位"]],
    },
    {
        "id": "unsupported_trip",
        "question": "修学旅行は北海道に決まったと記録されていますか。",
        "expect_rejection": True,
    },
    {
        "id": "unsupported_birthday",
        "question": "講師の誕生日は4月10日ですか。",
        "expect_rejection": True,
    },
]

REJECTION_PHRASES = (
    "確認できません",
    "記録からは",
    "記録には",
    "記載されていません",
    "わかりません",
    "分かりません",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lesson-folder", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def evaluate_answer(case: dict, answer: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if len(answer) > 200:
        failures.append("over_200_chars")
    if case.get("expect_rejection"):
        if not any(phrase in answer for phrase in REJECTION_PHRASES):
            failures.append("missing_rejection")
    for group in case.get("required_groups", []):
        if not any(term in answer for term in group):
            failures.append("missing:" + "|".join(group))
    return not failures, failures


def run_model(model_path: Path, lesson_folder: Path) -> dict:
    thread_count = select_asr_threads("file").num_threads
    load_start = time.perf_counter()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=4096,
        n_threads=thread_count,
        n_batch=256,
        verbose=False,
    )
    load_seconds = time.perf_counter() - load_start
    context = load_context(lesson_folder)
    results: list[dict] = []

    for case in CASES:
        messages = build_initial_messages(context)
        excerpts = find_relevant_transcript_excerpts(
            lesson_folder,
            case["question"],
        )
        first_chunk_at: list[float | None] = [None]
        started = time.perf_counter()

        def on_chunk(_text: str) -> None:
            if first_chunk_at[0] is None:
                first_chunk_at[0] = time.perf_counter()

        answer, _ = chat_one_turn(
            llm,
            messages,
            case["question"],
            relevant_excerpts=excerpts,
            on_chunk=on_chunk,
        )
        finished = time.perf_counter()
        passed, failures = evaluate_answer(case, answer)
        results.append({
            "id": case["id"],
            "question": case["question"],
            "answer": answer,
            "answer_chars": len(answer),
            "excerpt_chars": len(excerpts),
            "first_display_seconds": round(
                (first_chunk_at[0] or finished) - started,
                3,
            ),
            "total_seconds": round(finished - started, 3),
            "passed": passed,
            "failures": failures,
        })

    first_display_times = [item["first_display_seconds"] for item in results]
    total_times = [item["total_seconds"] for item in results]
    summary = {
        "model": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "threads": thread_count,
        "load_seconds": round(load_seconds, 3),
        "cold_first_display_with_load_seconds": round(
            load_seconds + first_display_times[0],
            3,
        ),
        "warm_first_display_median_seconds": round(
            statistics.median(first_display_times[1:]),
            3,
        ),
        "total_median_seconds": round(statistics.median(total_times), 3),
        "passed": sum(item["passed"] for item in results),
        "cases": len(results),
        "results": results,
    }
    del llm
    gc.collect()
    return summary


def main() -> int:
    args = parse_args()
    report = {
        "lesson_folder": str(args.lesson_folder),
        "models": [
            run_model(model.resolve(), args.lesson_folder.resolve())
            for model in args.model
        ],
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
