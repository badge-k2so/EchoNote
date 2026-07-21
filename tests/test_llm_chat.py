import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from otoweave_app.llm_chat import (
    _fit_inference_messages,
    build_initial_messages,
    build_retrieval_query,
    chat_one_turn,
    count_tokens,
    find_relevant_transcript_excerpts,
    load_context,
    summarize_llm_profile,
)
from scripts.production.template_summarize import (
    clean_llm_text,
    merge_part_records,
    replace_unexpected_scripts,
    sanitize_template_record,
    split_batches_by_tokens,
    strip_part_meta_notes,
    template_prompt,
)


class FakeTokenizerLlm:
    """Counts one token per character, with a configurable context size."""

    def __init__(self, n_ctx: int = 4096) -> None:
        self._n_ctx = n_ctx

    def n_ctx(self) -> int:
        return self._n_ctx

    @staticmethod
    def tokenize(value: bytes, add_bos: bool = False, special: bool = True):
        del add_bos, special
        return list(range(len(value.decode("utf-8"))))


class TranscriptRetrievalTests(unittest.TestCase):
    @staticmethod
    def write_transcript(folder: Path, texts: list[str]) -> None:
        segments = [
            {
                "id": f"seg_{index + 1:04d}",
                "start": index * 10.0,
                "end": (index + 1) * 10.0,
                "text": text,
            }
            for index, text in enumerate(texts)
        ]
        (folder / "transcript.json").write_text(
            json.dumps({"segments": segments}, ensure_ascii=False),
            encoding="utf-8",
        )

    def test_japanese_question_selects_relevant_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.write_transcript(folder, [
                "算数では分数の計算を練習しました。",
                "合理的配慮として、タブレットで回答する方法を確認しました。",
                "昼休みの予定について話しました。",
            ])

            excerpts = find_relevant_transcript_excerpts(
                folder,
                "合理的配慮について何と言っていましたか。",
            )

            self.assertIn("タブレットで回答", excerpts)
            self.assertNotIn("分数の計算", excerpts)
            self.assertNotIn("昼休み", excerpts)

    def test_english_question_selects_relevant_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.write_transcript(folder, [
                "We discussed the weather.",
                "Brain drain can cause a shortage of skilled workers.",
                "The class finished with a short quiz.",
            ])

            excerpts = find_relevant_transcript_excerpts(
                folder,
                "What did the recording say about brain drain?",
            )

            self.assertIn("shortage of skilled workers", excerpts)
            self.assertNotIn("weather", excerpts)

    def test_generic_question_uses_summary_without_arbitrary_excerpt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.write_transcript(folder, ["最初の話題です。", "次の話題です。"])

            excerpts = find_relevant_transcript_excerpts(
                folder,
                "内容を教えてください。",
            )

            self.assertEqual(excerpts, "")

    def test_separate_partial_matches_do_not_look_like_phrase_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.write_transcript(folder, [
                "保護者が支援を希望しています。",
                "本人から申し入れがありました。",
            ])

            excerpts = find_relevant_transcript_excerpts(folder, "本人の希望")

            self.assertEqual(excerpts, "")

    def test_short_follow_up_reuses_previous_question_for_retrieval(self) -> None:
        messages = build_initial_messages("記録")
        messages.extend([
            {"role": "user", "content": "合理的配慮には何がありましたか。"},
            {"role": "assistant", "content": "タブレットの利用です。"},
        ])

        query = build_retrieval_query(messages, "それはなぜ？")

        self.assertIn("合理的配慮", query)
        self.assertIn("それはなぜ", query)

    def test_excerpt_output_respects_character_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.write_transcript(folder, ["読み書き支援" * 200] * 5)

            excerpts = find_relevant_transcript_excerpts(
                folder,
                "読み書き支援について教えてください。",
                max_chunks=4,
                max_chars=500,
            )

            self.assertLessEqual(len(excerpts), 500)


class ChatPromptTests(unittest.TestCase):
    def test_persistent_context_uses_summary_not_partial_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            postprocess = folder / "postprocess"
            postprocess.mkdir()
            (postprocess / "school_record.md").write_text("全体要約", encoding="utf-8")
            (postprocess / "partial_records.md").write_text(
                "長い部分記録",
                encoding="utf-8",
            )

            context = load_context(folder)

            self.assertIn("全体要約", context)
            self.assertNotIn("長い部分記録", context)

    def test_retrieved_excerpt_is_inference_only(self) -> None:
        class FakeLlm:
            def __init__(self) -> None:
                self.messages = []

            @staticmethod
            def n_vocab() -> int:
                return 300000

            def create_chat_completion(self, **kwargs):
                self.messages = kwargs["messages"]
                return {
                    "choices": [
                        {"message": {"content": "記録ではタブレットを使うと説明されています。"}}
                    ]
                }

        llm = FakeLlm()
        messages = build_initial_messages("要約")
        answer, updated = chat_one_turn(
            llm,
            messages,
            "どんな支援ですか。",
            relevant_excerpts="[01:00-01:10] タブレットを使います。",
        )

        self.assertIn("タブレット", llm.messages[-1]["content"])
        self.assertEqual(updated[-2]["content"], "どんな支援ですか。")
        self.assertNotIn("transcript_excerpts", updated[-2]["content"])
        self.assertIn("タブレット", answer)

    def test_streaming_hides_thinking_and_emits_answer_chunks(self) -> None:
        class FakeStreamingLlm:
            @staticmethod
            def n_vocab() -> int:
                return 300000

            @staticmethod
            def create_chat_completion(**kwargs):
                if not kwargs.get("stream"):
                    raise AssertionError("streaming was not requested")
                values = ["<thi", "nk>内部メモ", "</think>記録", "の回答です。"]
                return iter([
                    {"choices": [{"delta": {"content": value}}]}
                    for value in values
                ])

        chunks: list[str] = []
        answer, _ = chat_one_turn(
            FakeStreamingLlm(),
            build_initial_messages("要約"),
            "質問",
            on_chunk=chunks.append,
        )

        self.assertEqual(answer, "記録の回答です。")
        self.assertEqual("".join(chunks), answer)
        self.assertNotIn("内部メモ", "".join(chunks))


class TokenBudgetTests(unittest.TestCase):
    def test_count_tokens_uses_model_tokenizer(self) -> None:
        llm = FakeTokenizerLlm()
        self.assertEqual(count_tokens(llm, "あいうえお"), 5)

    def test_count_tokens_falls_back_to_characters(self) -> None:
        self.assertEqual(count_tokens(SimpleNamespace(), "あいうえお"), 5)

    def test_small_prompt_is_left_unchanged(self) -> None:
        llm = FakeTokenizerLlm(n_ctx=4096)
        messages = build_initial_messages("短い要約")
        result = _fit_inference_messages(
            llm, messages, "質問です。", "抜粋" * 50, max_tokens=256
        )
        self.assertEqual(len(result), len(messages) + 1)
        self.assertIn("transcript_excerpts", result[-1]["content"])

    def test_oversized_excerpts_are_dropped_before_history(self) -> None:
        llm = FakeTokenizerLlm(n_ctx=1024)
        messages = build_initial_messages("要約")
        messages.extend([
            {"role": "user", "content": "前の質問"},
            {"role": "assistant", "content": "前の回答"},
        ])
        result = _fit_inference_messages(
            llm, messages, "質問です。", "長い抜粋" * 2000, max_tokens=256
        )
        # The oversized excerpt must shrink or disappear, history stays.
        self.assertLess(len(result[-1]["content"]), 4000)
        self.assertIn({"role": "user", "content": "前の質問"}, result)

    def test_oversized_history_is_dropped_oldest_first(self) -> None:
        llm = FakeTokenizerLlm(n_ctx=1024)
        messages = build_initial_messages("要約")
        for index in range(4):
            messages.append({"role": "user", "content": f"質問{index}" + "あ" * 400})
            messages.append({"role": "assistant", "content": f"回答{index}" + "い" * 400})
        result = _fit_inference_messages(
            llm, messages, "新しい質問", "", max_tokens=256
        )
        contents = [str(message.get("content", "")) for message in result]
        self.assertFalse(any(content.startswith("質問0") for content in contents))
        self.assertEqual(result[0]["role"], "system")
        total = sum(len(content) for content in contents)
        self.assertLessEqual(total, 1024)

    def test_summarize_profile_has_required_fields(self) -> None:
        profile = summarize_llm_profile()
        self.assertIn(profile["n_ctx"], (4096, 8192))
        self.assertGreater(profile["n_threads"], 0)
        self.assertGreater(profile["n_batch"], 0)
        self.assertGreater(profile["max_tokens_final"], 0)

    def test_summarize_model_is_4b_only_no_2b_fallback(self) -> None:
        # 2Bの要約は品質不足（A/B検証）のため、要約モデルは4B限定。
        # 低RAM機では None（=要約非対応）を返し、2Bへは切り替えない。
        from unittest.mock import patch as mock_patch

        from otoweave_app.llm_chat import find_summarize_model

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "models").mkdir()
            (root / "models" / "Qwen3.5-4B-Q4_K_M.gguf").write_bytes(b"x")
            (root / "models" / "Qwen3.5-2B-Q4_K_M.gguf").write_bytes(b"x")

            with mock_patch(
                "otoweave_app.llm_chat._total_physical_ram_bytes",
                return_value=4 * 1024**3,
            ):
                low_ram_choice = find_summarize_model(root)
            with mock_patch(
                "otoweave_app.llm_chat._total_physical_ram_bytes",
                return_value=16 * 1024**3,
            ):
                high_ram_choice = find_summarize_model(root)

        self.assertIsNone(low_ram_choice)
        self.assertIn("4B", high_ram_choice.name)


class TemplateSummarizeBudgetTests(unittest.TestCase):
    def test_split_batches_by_tokens_respects_budget(self) -> None:
        llm = FakeTokenizerLlm()
        paragraphs = ["段落" * 100 for _ in range(10)]
        batches = split_batches_by_tokens(llm, ["\n\n".join(paragraphs)], 500)
        self.assertGreater(len(batches), 1)
        for batch in batches:
            self.assertLessEqual(count_tokens(llm, batch), 500)

    def test_split_batches_hard_splits_single_huge_paragraph(self) -> None:
        llm = FakeTokenizerLlm()
        batches = split_batches_by_tokens(llm, ["あ" * 3000], 500)
        self.assertGreater(len(batches), 1)
        for batch in batches:
            self.assertLessEqual(count_tokens(llm, batch), 500)

    def test_merge_part_records_reduces_many_parts_hierarchically(self) -> None:
        calls: list[int] = []

        class FakeMergeLlm(FakeTokenizerLlm):
            @staticmethod
            def n_vocab() -> int:
                return 300000

            def create_chat_completion(self, **kwargs):
                prompt = kwargs["messages"][-1]["content"]
                calls.append(len(prompt))
                return {
                    "choices": [
                        {
                            "message": {"content": "## 要約\n- 統合結果"},
                            "finish_reason": "stop",
                        }
                    ]
                }

        llm = FakeMergeLlm(n_ctx=2048)
        args = SimpleNamespace(n_ctx=2048, max_tokens_final=400)
        parts = ["## 要約\n- " + f"内容{index}" + "あ" * 500 for index in range(6)]
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "log.txt"
            final, truncated = merge_part_records(
                llm, parts, "lesson_record", args, log_path
            )
        self.assertIn("統合結果", final)
        self.assertFalse(truncated)
        self.assertGreaterEqual(len(calls), 2)
        # Every merge prompt must fit the context window.
        for prompt_chars in calls:
            self.assertLessEqual(prompt_chars, 2048)

    def test_merge_part_records_strips_meta_notes_before_prompting(self) -> None:
        # A/Bベンチで9Bのpart記録にたまに混ざる「（入力内容に...記載なし）」
        # のようなメタ注記は、マージプロンプトに渡る前に除去されるべき
        # （渡すと「記載なし」自体を要約結果として書いてしまう）。
        prompts: list[str] = []

        class FakeMergeLlm(FakeTokenizerLlm):
            @staticmethod
            def n_vocab() -> int:
                return 300000

            def create_chat_completion(self, **kwargs):
                prompts.append(kwargs["messages"][-1]["content"])
                return {
                    "choices": [
                        {
                            "message": {"content": "## 議題\n- 統合結果"},
                            "finish_reason": "stop",
                        }
                    ]
                }

        llm = FakeMergeLlm(n_ctx=4096)
        args = SimpleNamespace(n_ctx=4096, max_tokens_final=400)
        parts = [
            "## 議題\n- 文化祭の日程\n- （入力内容に明示的な決定事項は記載なし）",
            "## 議題\n- - 該当なし\n- 予算の配分",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "log.txt"
            final, _truncated = merge_part_records(
                llm, parts, "meeting_memo", args, log_path
            )
        for prompt in prompts:
            self.assertNotIn("入力内容", prompt)
            self.assertNotIn("- - 該当なし", prompt)
        self.assertIn("統合結果", final)


class MergeStageNoOmissionInstructionTests(unittest.TestCase):
    """A/Bベンチで判明した9Bのマージ時自発的圧縮への対策。

    マージ系のstage（"merge round ..." / "final integration"）のときだけ
    「省略しない」指示をプロンプトに追記する。part段階の指示は変えない
    （instruction文字列そのものを書き換えるのではなく、stage判定で追記する
    実装であることを固定する）。
    """

    def test_part_stage_has_no_omission_instruction(self) -> None:
        prompt = template_prompt("meeting_record", "本文", "part 1")
        self.assertNotIn("一切省略せず", prompt)

    def test_merge_round_stage_adds_no_omission_instruction(self) -> None:
        prompt = template_prompt("meeting_record", "本文", "merge round 1")
        self.assertIn("一切省略せず", prompt)
        self.assertIn("短くまとめ直すことは禁止", prompt)

    def test_final_integration_stage_adds_no_omission_instruction(self) -> None:
        prompt = template_prompt("meeting_record", "本文", "final integration")
        self.assertIn("一切省略せず", prompt)
        self.assertIn("短くまとめ直すことは禁止", prompt)

    def test_original_template_instruction_is_untouched(self) -> None:
        # The addition must be appended, not a rewrite of the template's
        # own instruction text (so part-stage prompts stay unaffected).
        from scripts.production.template_summarize import TEMPLATES

        original_instruction = TEMPLATES["meeting_record"].get(
            "instruction", "指定された見出しに沿って整理する"
        )
        template_prompt("meeting_record", "本文", "final integration")
        self.assertEqual(
            TEMPLATES["meeting_record"].get(
                "instruction", "指定された見出しに沿って整理する"
            ),
            original_instruction,
        )


class PartMetaNoteFilterTests(unittest.TestCase):
    def test_strips_input_content_meta_note_line(self) -> None:
        text = "## 決定事項\n- （入力内容に明示的な決定事項は記載なし）\n- 実際の決定"
        result = strip_part_meta_notes(text)
        self.assertNotIn("入力内容", result)
        self.assertIn("実際の決定", result)

    def test_strips_garbled_duplicate_na_line(self) -> None:
        text = "## 未決事項\n- - 該当なし\n- 会場の割り当て"
        result = strip_part_meta_notes(text)
        self.assertNotIn("- - 該当なし", result)
        self.assertIn("会場の割り当て", result)

    def test_keeps_normal_na_placeholder(self) -> None:
        text = "## 未決事項\n- 該当なし"
        self.assertEqual(strip_part_meta_notes(text), text)

    def test_keeps_unrelated_parenthetical_content(self) -> None:
        text = "## 決定事項\n- 体育館を使用する（1階のみ）"
        self.assertEqual(strip_part_meta_notes(text), text)

    def test_sanitize_template_record_also_strips_freshly_generated_garbled_line(
        self,
    ) -> None:
        # Merge-smoke finding: the model can emit the "- - 該当なし" garble
        # itself while merging (not only inherit it from part inputs), so
        # ensure_template_sections's sanitize_template_record pass must
        # also catch it, or it survives into the final record.
        text = "## 決まったこと\n- - 該当なし\n\n## 次にやること\n- 会場を予約する"
        result = sanitize_template_record(text)
        self.assertNotIn("- - 該当なし", result)
        self.assertIn("会場を予約する", result)


class UnexpectedScriptTests(unittest.TestCase):
    """Qwen occasionally leaks Cyrillic etc. mid-word (e.g. 「ファкультイ」)."""

    def test_cyrillic_run_becomes_unknown_marker(self):
        # [不明] is picked up by build_review_flags, so the garbled word
        # surfaces in the 要確認リスト instead of shipping silently.
        self.assertEqual(
            replace_unexpected_scripts("英語ファкультイに参加した"),
            "英語ファ[不明]イに参加した",
        )

    def test_japanese_english_and_greek_stay_untouched(self):
        text = "数学のβ版テスト: Reading & Writing 支援（第2回）"
        self.assertEqual(replace_unexpected_scripts(text), text)

    def test_clean_llm_text_applies_replacement(self):
        self.assertEqual(
            clean_llm_text("<think>草案</think>안녕テスト"),
            "[不明]テスト",
        )


if __name__ == "__main__":
    unittest.main()
