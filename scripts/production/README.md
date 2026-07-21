# scripts/production

Scripts the running app and the distribution build actually invoke as
subprocesses: `template_summarize.py` and `school_hybrid_postprocess.py`
(called from `otoweave_app/llm_chat.py`), `record_filename.py` (imported by
`otoweave_app/transcription_service.py`), and `windows_audio_file_dialog.ps1`
(called from `otoweave_app/otoweave_app.py`). The `run_*.ps1` wrappers here
let you run the same postprocessing scripts standalone from the command
line, e.g.:

```powershell
.\scripts\production\run_school_hybrid_postprocess.ps1 -TranscriptFile ".\runs\...\output\raw_transcript.txt"
```

Do not move or rename files in this folder without updating the references
above and in `tests/`. See `PROJECT_RECORD.md` for how these scripts were
selected over the alternatives kept in `scripts/prototypes/` and
`scripts/benchmarks/`.
