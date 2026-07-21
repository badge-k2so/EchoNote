# OtoWeave Development Notes

Local-first Windows transcription research project and functional OtoWeave
app for reducing note-taking load in class. Audio, transcripts, marks, and
lesson metadata remain on the device. The app does not call cloud AI APIs,
telemetry, analytics, or automatic sync services.

旧称: EchoNote（2026-07-20に改名）。開発初期のコード名は LearningAccessWhisperTest。

## Try It (git clone Quick Setup)

Anyone with a Windows 10/11 PC and an internet connection can try OtoWeave
directly from a clone, no separately-shared package required:

```powershell
git clone https://github.com/badge-k2so/otoweave.git
cd otoweave
.\distribution\setup_easy.ps1
```

Run it in PowerShell (right-click the folder → "Open in Terminal", or open
PowerShell and `cd` into the cloned folder first). The script checks for
Python 3.12 (installing it via `winget` if missing), creates a `.venv`,
installs the required libraries, checks this PC's RAM/disk space, then
downloads the AI models it needs (Japanese ASR, English ASR, language
routing, and the AI chat model always; the larger AI-summary model only on
machines with more than 11.5 GB RAM). The first run downloads about 2.7 GB
(about 5.3 GB if the AI-summary model is included) and can take a while
depending on your connection; add `-Lite` to skip the AI-summary model
regardless of RAM. If a download is interrupted, run the script again — it
resumes and skips whatever already finished. When it's done, start the app
with:

```powershell
.\run_otoweave.ps1
```

This is a separate, simpler path than the tester distribution package below;
both are kept side by side. `setup_easy.ps1` is for anyone with internet
access; the prototype-test package below is for offline test machines.

## OtoWeave Windows App (CustomTkinter UI)

The desktop app (`otoweave_app/otoweave_app.py`) uses a four-pane
CustomTkinter layout: an activity bar (録音 / 取り込み / ノート / 補正辞書 /
設定), a note list, the transcript body, and a summary + AI tutor pane.
It provides:

- Microphone recording for in-person use and Windows WASAPI loopback capture
  for Zoom or Google Meet playback audio, with a 3-second input level test.
- 16 kHz mono audio with timeline-preserving energy VAD. Short utterances
  are kept: pause/stop flushes emit even sub-second speech.
- Japanese live ASR with ReazonSpeech K2, English live ASR with Parakeet TDT
  int8, and a recording-only mode. Live text follows the newest line; when
  the user scrolls up a `↓ 最新へ戻る` button appears instead of forcing
  the view down.
- Important (`★`), unclear (`?`), and question (`!`) marks.
- A built-in player for saved lessons (play/pause, position, duration).
  Clicking a timestamp in the transcript plays from that position, so a
  hard-to-read section can be listened to instead.
- Text-to-speech for the transcript and the AI summary (`🔊 読み上げ`),
  using the Windows built-in Japanese voice (System.Speech / Haruka):
  fully offline, no extra model or dependency, press again to stop.
- Display settings for font (Japanese UD fonts prioritized, OpenDyslexic
  offered when installed), text size, line spacing, reading width,
  light/dark mode, live follow, and current-utterance highlight.
- AI summaries (**beta** — labeled as β版 in the UI, with a permanent
  reminder to verify against the transcript and audio) with selectable and
  user-editable templates, generated fully locally by a Qwen GGUF model in
  a cancellable subprocess. The context size scales down automatically on
  low-RAM machines, long transcripts are merged hierarchically, and
  truncated output is flagged in the record.
- A local AI tutor chat about the selected transcript (resident 2B model,
  token-budgeted prompt).
- A user correction dictionary applied to recognition output and provided to
  the summary model as reference vocabulary (aliases must be 2+ characters).
- The note list is built from `metadata.json` only, so startup stays fast
  with hundreds of lessons; transcript bodies load on selection and body
  search runs in the background.
- Import existing OGG, WAV, MP3, M4A, Opus, FLAC, AAC, or WMA recordings via
  the Windows file dialog or drag & drop. The source file is never modified.
- Run Japanese, English, low-memory mixed, or optional Qwen3-ASR-1.7B
  transcription later on an audio-only saved lesson.
- Edit the transcript as one document; the before/after history is kept in
  `corrections.jsonl` (append-only).
- Deleting a lesson moves it to `_trash` inside the storage folder; trash
  entries are purged automatically after 30 days.

Install the small audio dependencies once:

```powershell
cd C:\whisper
.\.venv\Scripts\python.exe -m pip install -r requirements_mvp.txt
.\.venv\Scripts\python.exe -m pip install customtkinter tkinterdnd2
```

(`distribution\requirements_dist.txt` pins the exact versions used for the
tester package, including these two.)

Start the app:

```powershell
.\run_otoweave.ps1
```

Start with a sample lesson for UI review:

```powershell
.\run_otoweave.ps1 -Demo -DataRoot ".\runs\learning_access_demo"
```

The `English S&E` sample includes a short English WAV generated locally with
Windows speech synthesis, so playback and later Parakeet retranscription can be
tested. Older copies of this sample that lack audio are repaired on startup.

Lessons are stored under `Documents\OtoWeave` by default. Each completed
lesson contains:

```text
audio.opus
transcript.json
transcript.json.bak   (previous good version, crash recovery)
transcript.md
marks.json
metadata.json
corrections.jsonl     (only after manual corrections)
postprocess\          (only after AI summaries)
```

All lesson files are written atomically with fsync; if `transcript.json` is
damaged the app falls back to `transcript.json.bak`, and unreadable lessons
stay visible in the list marked `⚠ 要修復` instead of disappearing.

For an online lesson, choose `PC音声` as the recording target. This captures
the sound played by Windows, including the remote participant and shared
media. It does not identify speakers or label text as teacher/student. The
app does not mix the local microphone with PC playback; choose `マイク` when
the local room is the audio source.

For a lesson created without transcription, select it and press
`文字起こしを開始`. Choose Japanese (ReazonSpeech), the standard low-memory
mixed mode, or English (Parakeet). Qwen3-ASR-1.7B is offered only when its
local model files are installed; it is intended for systems with at least
16 GB RAM. If a transcript already exists, the app asks before replacing its
text; `★` and `?` marks are transferred to the nearest new timestamps. A
failed retry leaves the existing transcript unchanged.

The standard mixed mode follows this low-memory pipeline:

```text
VAD chunks
  -> SpeechBrain Japanese/English classification
  -> classify Japanese and English chunks
  -> ReazonSpeech batch (Japanese)
  -> release ReazonSpeech
  -> Parakeet batch (English)
  -> release Parakeet
  -> restore timestamp order
```

Low-confidence language decisions are marked unclear. Very short or close-score
chunks default to Japanese because the primary use case is Japanese school
audio. Per-chunk decisions and confidence scores are saved in
`logs\language_routing.json`.

`Models & Licenses` lists the installed recognition models, their purpose,
license, availability, and official distribution page. This includes
ReazonSpeech K2 v2 (Apache-2.0), NVIDIA Parakeet TDT 0.6B v2 int8 (CC-BY-4.0),
SpeechBrain ECAPA-TDNN VoxLingua107 (Apache-2.0), optional Qwen3-ASR-1.7B
(Apache-2.0), and the sherpa-onnx runtime (Apache-2.0).

Use `編集` above the transcript to correct recognition errors and press
`保存`. Each change updates the transcript and appends a local correction
record (timestamp, segment ID, original text, corrected text) to
`corrections.jsonl`.

## Prototype-Test Distribution

`distribution\build_distribution.ps1` assembles a portable test package
(about 6 GB) under `dist\OtoWeave_ProtoTest_<date>\` containing the app
code, only the production models (see `distribution\docs\モデル構成.md`),
the ReazonSpeech HF cache, offline wheels, ffmpeg, setup/launch scripts,
and the test protocol documents:

```powershell
.\distribution\build_distribution.ps1                 # requires internet (wheels)
.\distribution\build_distribution.ps1 -Lite           # 64GB-SSD machines: no 4B model (~2.9 GB)
.\distribution\build_distribution.ps1 -IncludePythonInstaller  # bundle Python 3.12
```

The Lite build omits Qwen3.5-4B; the app then uses 2B for summaries
automatically (which is also what happens on <7 GB-RAM machines). Total
footprint after setup: standard ~7 GB, Lite ~4.5 GB.

On the test PC (offline OK): copy the folder to a local disk, run
`setup_test_pc.ps1` (creates a venv, installs wheels offline, runs
`verify_setup.py`), then double-click `OtoWeaveを起動.bat`. Model roles
are fixed as: ReazonSpeech K2 = Japanese ASR, Parakeet = English ASR,
SpeechBrain = language routing, Qwen3.5-4B = summaries (auto-switches to
2B on <7 GB RAM), Qwen3.5-2B = chat, Windows Haruka = TTS.

### AI Summary — Higher-Tier Model (9B, Optional for 16 GB-RAM Machines)

`Qwen3.5-9B-Q4_K_M.gguf` (5.29 GiB) is not bundled in the distribution
package by default. On machines with more than 15 GB RAM the app
automatically prefers it over the 4B model for summaries when the file is
present at `models\Qwen3.5-9B-Q4_K_M.gguf`; at or below 15 GB it is not
used and the app falls back to 4B. To add it after setup, run
`distribution\download_9b_model.ps1` on the test machine — it checks RAM
and free disk space, downloads with resume support via `curl.exe`, and
verifies the downloaded file is at least 5 GB before finishing.
`setup_test_pc.ps1` already removes the 9B file automatically on machines
at or below the 15 GB threshold.

### Speaker Diarization (Prototype)

`scripts\prototypes\diarization_prototype.py` is a standalone research script — not
wired into the app yet — that runs sherpa-onnx offline diarization
(pyannote segmentation + 3D-Speaker embedding, `models\diarization\`,
about 46 MB total, bundled by `build_distribution.ps1`) over a saved
recording. Measured on an i5-1145G7 with 4 threads: RTF 0.174 (a
71-minute recording processed in about 12.4 minutes), peak memory about
580 MB. Lower-spec machines (e.g. Celeron-class GIGA School devices) are
expected to be slower (roughly 2-4x has been estimated but not yet
measured on such hardware).

## Folder Layout

```text
whisper/
├─ README.md
├─ otoweave_app/
├─ run_otoweave.ps1
├─ requirements_mvp.txt
├─ scripts/
│  ├─ production/    (used by the app: template_summarize.py, etc.)
│  ├─ benchmarks/    (ASR/LLM comparison scripts)
│  ├─ prototypes/    (pre-app experiments, incl. run_whisper_test.ps1)
│  └─ tools/
├─ engines/
│  ├─ ffmpeg/
│  │  └─ ffmpeg.exe
│  └─ whisper/
│     ├─ whisper-cli.exe
│     └─ models/
│        └─ ggml-base.bin
└─ runs/
```

## whisper.cpp Prototype Runner (`scripts\prototypes\run_whisper_test.ps1`)

The sections below document the original whisper.cpp-based prototype runner,
kept in `scripts\prototypes\` as a comparison baseline; it is not part of the
current app pipeline (ReazonSpeech K2 + Parakeet, see above).

## Setup

Place these files before running the script:

- `engines\ffmpeg\ffmpeg.exe`
- `engines\whisper\whisper-cli.exe`
- `engines\whisper\models\ggml-base.bin`

The default input file is:

```text
C:\Users\<your-user-name>\Downloads\2026-05-22 13_30_39.ogg
```

You can pass another input path with `-InputFile`.

## Run First 10 Minutes Only

```powershell
cd C:\whisper
.\scripts\prototypes\run_whisper_test.ps1 -TEST_ONLY:$true
```

Or with an explicit input file:

```powershell
.\scripts\prototypes\run_whisper_test.ps1 -InputFile "C:\Users\<your-user-name>\Downloads\2026-05-22 13_30_39.ogg" -TEST_ONLY:$true
```

Model can be selected with `-Model base`, `-Model small`, or `-Model medium`:

```powershell
.\scripts\prototypes\run_whisper_test.ps1 -TEST_ONLY:$true -Model small
.\scripts\prototypes\run_whisper_test.ps1 -TEST_ONLY:$true -Model medium
```

## Run Full File

```powershell
cd C:\whisper
.\scripts\prototypes\run_whisper_test.ps1 -InputFile "C:\Users\<your-user-name>\Downloads\2026-05-22 13_30_39.ogg" -TEST_ONLY:$false
```

## What The Script Does

- Creates `runs\<timestamp>\`.
- Creates `chunks`, `output`, and `logs` folders.
- Uses ffmpeg to convert audio to 16 kHz mono 16-bit PCM WAV.
- Splits audio into 10-minute chunks.
- If `TEST_ONLY` is `true`, ffmpeg only processes the first 10 minutes.
- Runs `whisper-cli.exe` sequentially, never in parallel.
- Uses Japanese language mode with `-l ja`.
- Creates per-chunk `txt` and `srt` outputs.
- Creates `output\full_transcript.txt`.
- Saves ffmpeg timing, per-chunk whisper timing, total timing, and errors in `logs\log.txt`.
- Samples RAM usage every 0.5 seconds into `logs\memory.csv`.
- Logs peak system RAM usage and peak ffmpeg / whisper process working set values.

## Output

Each run is saved in:

```text
runs\<timestamp>\
```

Important files:

- `logs\log.txt`
- `logs\memory.csv`
- `logs\ffmpeg.txt`
- `logs\chunk_0000.whisper.txt`
- `output\chunk_0000.txt`
- `output\chunk_0000.srt`
- `output\full_transcript.txt`

## Recording Filename Suggestions

The ReazonSpeech, Parakeet, and hybrid runners create
`output\filename_suggestion.json` after transcription. The suggested format is:

```text
YYYY-MM-DD_short-content-title.ogg
```

The recording date is taken from an explicit recording timestamp when available,
then from the original filename, and finally from the file creation time. A short
title is preserved from a descriptive original filename or extracted from the
transcript using deterministic topic-term rules. No LLM is required.

This file is only a suggestion for the future GUI. The original recording is not
renamed automatically, and `requires_user_confirmation` is always `true` so the
user can review or edit the name first.

## School Postprocess Fallback Types

`scripts\production\run_school_hybrid_postprocess.ps1` writes fallback and quality-gate state to
`output\school_postprocess_summary.json`.

There are two different fallback meanings:

- `technical fallback`
  - Recorded as `final_used_fallback: true`.
  - Used only when the LLM output is empty, malformed, missing the required format,
    or the LLM step fails technically.
  - This means the final record could not be trusted as an LLM-generated final.

- `learner quality fallback`
  - Recorded as `learner_final_quality_fallback: true`.
  - `final_used_fallback` remains `false` when the format was valid.
  - Used when the output format succeeded, but important learner-support content
    such as reading/writing difficulty, LD, learning disability, reasonable
    accommodation, ICT support,困りごと, or確認事項 was dropped.
  - In this case `final_generation_mode` is `deterministic_merge`.
  - The deterministic merge is adaptive: it uses dynamically extracted core
    terms from the recording, then falls back to generic learner-support checks
    instead of assuming every class is about reading/writing difficulty.

When the learner final format and content quality both pass, Python still checks
`## 次に先生や大人に確認すること`. If the LLM produced too few follow-up
questions, deterministic school-support questions are appended up to seven
items. The questions are designed for school settings broadly: lessons,
interviews, study sessions, support meetings, and staff meetings. This is
recorded as `next_checks_completed` and `next_checks_count` in
`school_postprocess_summary.json`.

Important summary fields:

```json
{
  "final_used_fallback": false,
  "learner_final_quality_fallback": true,
  "format_check_passed": true,
  "content_quality_check_passed": false,
  "quality_gate_reason": "missing_core_topic_terms",
  "missing_core_terms": ["読み書き障害", "ICT支援"],
  "final_generation_mode": "deterministic_merge"
}
```

## Learner PC Postprocess Policy

For GPU-less learner PCs, the standard mode uses Qwen3.5 4B Q4 plus
deterministic Python safeguards. The LLM should do short rewriting,
classification, and section formatting; Python should decide candidate selection,
core-topic scoring, noisy-term filtering, and quality gates.

`part_summaries.md` therefore uses the neutral heading
`## この部分に出てきた内容` instead of asking the model to decide
`重要事項`. Suspicious ASR fragments, long meaningless katakana terms, and
unclear noun-only lines are moved to `## 要確認箇所` so they do not become
learner-facing "important" content. The old `## 重要事項` heading is still read
for compatibility with older runs.

Recommended CPU-only profile:

- standard: Qwen3.5 4B Q4 + dynamic core terms + deterministic merge + quality gate
- avoid: 9B long-document final integration on low-spec CPU-only machines

## Notes

The script expects the current whisper.cpp CLI arguments:

```powershell
whisper-cli.exe -m <model> -f <wav> -l ja -otxt -osrt -of <output-base>
```

If your `whisper-cli.exe` build uses different arguments, adjust the `$whisperArgs` section in `scripts\prototypes\run_whisper_test.ps1`.
