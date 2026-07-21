# scripts/prototypes

Experiments that ran before the current app design (ReazonSpeech K2 +
Parakeet + Qwen GGUF postprocess, driven from `otoweave_app/`), kept as
evidence of what was tried and why it was not adopted, not as supported
tools. Example of running one directly:

```powershell
.\scripts\prototypes\run_reazon_k2_vad_test.ps1 -InputFile "C:\recording.m4a" -MaxDurationSeconds 0
```

`diarization_prototype.py` is the one exception still referenced by the
distribution test protocol (see `distribution/docs/テスト手順書.md`
scenario 8) as an unintegrated, optional feature test.
`README_MOONSHINE_TEST.md` and `README_REAZON_TEST.md` document two of the
earlier engine trials. See `PROJECT_RECORD.md` for the full comparison
history and why ReazonSpeech K2 + Parakeet were chosen instead.
