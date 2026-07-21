# Moonshine Voice Japanese Test

## Purpose

This test measures Moonshine Voice with the Japanese model on the same first 10 minutes used for the whisper.cpp and ReazonSpeech comparisons.

This is only a CLI validation. No GUI app is created.

## Setup

```powershell
cd C:\whisper
.\setup_moonshine_voice.ps1
```

The setup script installs `moonshine-voice` into `.venv` and downloads the Japanese model.

## Run

```powershell
cd C:\whisper
.\run_moonshine_test.ps1
```

The script:

- converts the first 10 minutes to `16kHz mono 16-bit PCM WAV`
- splits that 10-minute WAV into 30-second chunks
- loads the Moonshine Japanese model once
- processes chunks sequentially in one Python process
- writes per-chunk transcripts, `full_transcript.txt`, `summary.json`, and logs

## Output

```text
runs/<timestamp>_moonshine_ja/
  source/
  chunks_30sec/
  output/
  logs/
```

## Notes

Moonshine Voice is optimized for low-latency voice applications. The Japanese model used by the Python downloader is the quantized Japanese base model.

For non-Latin languages, the upstream documentation recommends setting `max_tokens_per_second` higher to avoid cutting valid output. This test uses `max_tokens_per_second=13.0`.

Moonshine's license notice says the Japanese model is released under the non-commercial Moonshine Community License. Check licensing before any school or commercial deployment.
