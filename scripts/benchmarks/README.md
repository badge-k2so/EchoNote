# scripts/benchmarks

Standalone comparison and measurement scripts used to decide which ASR/LLM
engine or model size to use in production — not wired into the app. Example:

```powershell
.\.venv\Scripts\python.exe scripts\benchmarks\compare_japanese_asr.py
```

`run_qwen35_4b_postprocess_test.ps1` re-runs the production
`school_hybrid_postprocess.py` pipeline forcing the Qwen3.5-4B model, to
compare quality/speed against the default. See `PROJECT_RECORD.md` for the
measured results these scripts produced.
