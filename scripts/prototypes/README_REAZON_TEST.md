# ReazonSpeech-k2-v2 CPU/int8 Test

## Purpose

This test checks whether ReazonSpeech-k2-v2 is practical on a Windows 11 class PC without GPU acceleration.

The comparison target is the existing `whisper.cpp medium` test. `whisper.cpp medium` produced usable Japanese transcription quality, but CPU processing was slow, so this test measures ReazonSpeech-k2-v2 with:

- `device="cpu"`
- `precision="int8"`
- `language="ja"`

This is only a CLI validation. No GUI app is created.

## Target PC

Reference environment:

- Windows 11
- Intel Core i5-1145G7 class CPU
- RAM 16GB
- No GPU requirement

## Why 30-Second Chunks

ReazonSpeech-k2-v2 is intended for short audio clips. Long audio should not be passed to the model as one file. This test first cuts the same source audio down to the first 10 minutes, then splits it into 30-second WAV chunks.

The Python process loads the model once, then processes all 30-second chunks sequentially. This avoids reloading the model for every chunk and makes timing fairer.

## Setup

Run:

```powershell
cd C:\whisper
.\setup_reazon_k2.ps1
```

The setup script:

- checks Python
- creates `.venv`
- upgrades pip
- downloads the official ReazonSpeech GitHub source zip
- installs `pkg/k2-asr` into `.venv`
- verifies `reazonspeech.k2.asr` import

It does not install packages into global Python.

Initial setup may take time because packages and the model may be downloaded.

## Run

Run:

```powershell
cd C:\whisper
.\run_reazon_k2_test.ps1
```

The script processes only the first 10 minutes. It does not process the full 70-minute audio.

## Output

Each run creates:

```text
runs/<timestamp>_reazon_k2_int8/
  source/
    test_10min.wav
  chunks_30sec/
    chunk_000.wav
    chunk_001.wav
  output/
    chunk_000.txt
    full_transcript.txt
    summary.json
  logs/
    log.txt
    system_resources.csv
    ffmpeg_10min.txt
    ffmpeg_chunks.txt
    reazon_k2_stdout.txt
```

If a chunk fails, `output/chunk_XXX.error.txt` is created.

## Compare With whisper.cpp medium

Compare the ReazonSpeech run with:

```text
runs/20260522_224711_medium/
```

Comparison items:

- total processing time for the same 10-minute audio
- subjective transcription accuracy
- proper nouns
- Japanese naturalness
- sentence boundaries
- CPU usage
- RAM usage
- setup difficulty
- ease of embedding into a GUI app

## Notes

ReazonSpeech-k2-v2 may download model files on first use. Internet access may be required the first time.

For a future school-facing Surface app, Python dependency handling needs separate planning. Options may include bundling a local Python runtime, packaging the environment, or replacing the Python layer with a service boundary.

## Troubleshooting

If setup fails:

- check `runs/setup_reazon_k2/setup.log`
- confirm Python 3 is installed
- confirm network access
- rerun `.\setup_reazon_k2.ps1`

If transcription fails:

- check `logs/log.txt`
- check `logs/reazon_k2_stdout.txt`
- check any `output/chunk_XXX.error.txt`

Some Windows environments may need additional runtime dependencies depending on the ONNX package stack installed by `reazonspeech[k2]`.
