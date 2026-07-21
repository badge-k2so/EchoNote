"""Prototype: offline speaker diarization using sherpa-onnx.

Loads an audio file (any format librosa/soundfile can read), converts it to
16kHz mono, truncates to --max_seconds, and runs sherpa-onnx's
OfflineSpeakerDiarization (pyannote segmentation + 3D-Speaker embedding +
fast clustering) over it.

Outputs, under --output_dir:
  - segments.json  : [{"start": float, "end": float, "speaker": int}, ...]
  - segments.txt   : human-readable "HH:MM:SS-HH:MM:SS  speaker_N" lines

This is a standalone research/validation script for evaluating whether
diarization is worth wiring into transcription_service.py.  It does not
import or modify any existing otoweave_app code.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import sherpa_onnx

SAMPLE_RATE = 16000

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "diarization"
DEFAULT_SEGMENTATION_MODEL = (
    DEFAULT_MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
)
DEFAULT_EMBEDDING_MODEL = (
    DEFAULT_MODELS_DIR / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)


def format_hhmmss(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def load_audio_16k_mono(audio_path: Path, max_seconds: float) -> np.ndarray:
    duration = max_seconds if max_seconds and max_seconds > 0 else None
    samples, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True, duration=duration)
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"librosa.load returned sr={sr}, expected {SAMPLE_RATE}")
    return samples.astype(np.float32)


def build_diarizer(
    segmentation_model: Path,
    embedding_model: Path,
    num_threads: int,
    num_speakers: int,
    threshold: float,
) -> "sherpa_onnx.OfflineSpeakerDiarization":
    missing = [str(p) for p in (segmentation_model, embedding_model) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Diarization model files are missing: " + ", ".join(missing))

    segmentation_config = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=str(segmentation_model)
        ),
        num_threads=num_threads,
        debug=False,
        provider="cpu",
    )
    embedding_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(embedding_model),
        num_threads=num_threads,
        debug=False,
        provider="cpu",
    )
    clustering_config = sherpa_onnx.FastClusteringConfig(
        num_clusters=num_speakers,
        threshold=threshold,
    )
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=segmentation_config,
        embedding=embedding_config,
        clustering=clustering_config,
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("OfflineSpeakerDiarizationConfig failed validate()")
    return sherpa_onnx.OfflineSpeakerDiarization(config)


def progress_callback(num_processed_chunks: int, num_total_chunks: int) -> int:
    if num_total_chunks > 0 and num_processed_chunks % 10 == 0:
        pct = 100.0 * num_processed_chunks / num_total_chunks
        print(f"  ... {num_processed_chunks}/{num_total_chunks} chunks ({pct:.0f}%)", flush=True)
    return 0  # non-zero would abort processing


def main() -> int:
    parser = argparse.ArgumentParser(description="sherpa-onnx offline speaker diarization prototype")
    parser.add_argument("--audio", required=True, help="Path to input audio (any format librosa can read)")
    parser.add_argument("--output_dir", required=True, help="Directory to write segments.json / segments.txt")
    parser.add_argument("--num_speakers", type=int, default=-1, help="Fixed speaker count, -1 = auto (default)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Clustering distance threshold (used when num_speakers=-1)")
    parser.add_argument("--max_seconds", type=float, default=600.0, help="Only process the first N seconds of audio")
    parser.add_argument("--num_threads", type=int, default=2, help="ONNX Runtime intra-op threads for both models")
    parser.add_argument("--segmentation_model", default=str(DEFAULT_SEGMENTATION_MODEL), help="Path to pyannote segmentation model.onnx")
    parser.add_argument("--embedding_model", default=str(DEFAULT_EMBEDDING_MODEL), help="Path to speaker embedding .onnx")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Audio:            {audio_path}", flush=True)
    print(f"Max seconds:      {args.max_seconds}", flush=True)
    print(f"num_threads:      {args.num_threads}", flush=True)
    print(f"num_speakers:     {args.num_speakers}", flush=True)
    print(f"threshold:        {args.threshold}", flush=True)

    load_start = time.time()
    samples = load_audio_16k_mono(audio_path, args.max_seconds)
    load_elapsed = time.time() - load_start
    audio_seconds = len(samples) / SAMPLE_RATE
    print(f"Loaded {audio_seconds:.1f}s of audio at {SAMPLE_RATE}Hz mono in {load_elapsed:.2f}s", flush=True)

    diarizer = build_diarizer(
        segmentation_model=Path(args.segmentation_model),
        embedding_model=Path(args.embedding_model),
        num_threads=args.num_threads,
        num_speakers=args.num_speakers,
        threshold=args.threshold,
    )
    if diarizer.sample_rate != SAMPLE_RATE:
        raise RuntimeError(
            f"Diarizer expects sample_rate={diarizer.sample_rate}, but audio was loaded at {SAMPLE_RATE}"
        )

    print("Running diarization...", flush=True)
    process_start = time.time()
    result = diarizer.process(samples, callback=progress_callback)
    process_elapsed = time.time() - process_start

    real_time_factor = process_elapsed / audio_seconds if audio_seconds > 0 else float("nan")
    seconds_per_10min = process_elapsed * (600.0 / audio_seconds) if audio_seconds > 0 else float("nan")
    print(
        f"Diarization done in {process_elapsed:.2f}s "
        f"(RTF={real_time_factor:.3f}, ~{seconds_per_10min:.1f}s per 10min audio)",
        flush=True,
    )
    print(f"num_speakers detected: {result.num_speakers}", flush=True)
    print(f"num_segments: {result.num_segments}", flush=True)

    segments = result.sort_by_start_time()

    segments_data = [
        {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "speaker": int(seg.speaker),
        }
        for seg in segments
    ]
    segments_json_path = output_dir / "segments.json"
    segments_json_path.write_text(
        json.dumps(segments_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = []
    for seg in segments_data:
        lines.append(
            f"{format_hhmmss(seg['start'])} - {format_hhmmss(seg['end'])}  speaker_{seg['speaker']}"
        )
    segments_txt_path = output_dir / "segments.txt"
    segments_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "audio": str(audio_path),
        "audio_seconds_processed": round(audio_seconds, 3),
        "num_threads": args.num_threads,
        "num_speakers_arg": args.num_speakers,
        "threshold": args.threshold,
        "audio_load_seconds": round(load_elapsed, 3),
        "process_seconds": round(process_elapsed, 3),
        "real_time_factor": round(real_time_factor, 4),
        "seconds_per_10min_audio": round(seconds_per_10min, 2),
        "num_speakers_detected": result.num_speakers,
        "num_segments": result.num_segments,
    }
    meta_path = output_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {segments_json_path}", flush=True)
    print(f"Wrote: {segments_txt_path}", flush=True)
    print(f"Wrote: {meta_path}", flush=True)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
