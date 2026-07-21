# school_hybrid_format.md
# このファイルはscripts/production/run_school_hybrid_postprocess.ps1が要求するプレースホルダーです。
# 実際のプロンプトはscripts/production/school_hybrid_postprocess.pyのPART_USER_PROMPT / FINAL_USER_PROMPTに定義されています。

## PART_USER_PROMPT の設計方針（Qwen3.5-4B標準向け最適化）

- ルール数を絞る（小型CPUモデルは長い指示リストを追従しにくい）
- 最重要ルール「整形記録 にタイムスタンプを書かない」を明示
- 出力見出しを3つに絞る（整形記録 / 英語・専門語・固有名詞 / 要確認箇所）
- 矛盾する項目数指定を統一（最大6項目）
- 重要事項 と 3行サマリー はFINAL統合ステップのみで生成

## FINAL_USER_PROMPT の設計方針

- 統合ルール6つに絞る
- 出力見出しを4つ（整形記録 / 英語・専門語・固有名詞 / 要確認箇所 / 3行サマリー）
- タイムスタンプ禁止ルールを再掲
