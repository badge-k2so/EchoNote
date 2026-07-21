"""Split audio at natural silence boundaries using ffmpeg silencedetect.

Each output chunk is at most --max_seconds long (default 28).
Splits happen at the midpoint of the silence period closest to the
MAX_CHUNK_SECONDS boundary.  If no silence is found within a window,
a hard split is inserted at MAX_CHUNK_SECONDS.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def detect_silences(
    ffmpeg_exe: str,
    wav_path: str,
    noise: str,
    min_duration: float,
) -> list[tuple[float, float]]:
    result = subprocess.run(
        [
            ffmpeg_exe,
            "-i", wav_path,
            "-af", f"silencedetect=noise={noise}:d={min_duration}",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    output = result.stderr

    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", output)]
    ends   = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", output)]

    silences: list[tuple[float, float]] = []
    for s, e in zip(starts, ends):
        silences.append((s, e))

    # silencedetect may emit a start without a matching end if audio ends in silence
    if len(starts) > len(ends):
        silences.append((starts[-1], starts[-1]))  # treat as zero-length

    return silences


def get_duration(ffmpeg_exe: str, wav_path: str) -> float:
    result = subprocess.run(
        [ffmpeg_exe, "-i", wav_path],
        capture_output=True,
        text=True,
    )
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", result.stderr)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return 0.0


def compute_split_points(
    silences: list[tuple[float, float]],
    total_duration: float,
    max_seconds: float,
) -> list[float]:
    """Return a list of cut timestamps, starting with 0 and ending with total_duration."""
    midpoints = sorted((s + e) / 2 for s, e in silences)

    split_points = [0.0]
    while True:
        current_start = split_points[-1]
        hard_limit = current_start + max_seconds

        if hard_limit >= total_duration:
            break

        # Find the latest midpoint that is strictly within [current_start, hard_limit]
        candidates = [mp for mp in midpoints if current_start < mp <= hard_limit]
        if candidates:
            split_points.append(max(candidates))
        else:
            # No silence available — force a hard split
            split_points.append(hard_limit)

    split_points.append(total_duration)
    return split_points


def extract_chunks(
    ffmpeg_exe: str,
    wav_path: str,
    split_points: list[float],
    output_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        duration = split_points[i + 1] - start
        out_path = output_dir / f"chunk_{i:03d}.wav"

        subprocess.run(
            [
                ffmpeg_exe, "-y",
                "-i", wav_path,
                "-ss", f"{start:.6f}",
                "-t",  f"{duration:.6f}",
                "-ar", "16000",
                "-ac", "1",
                "-sample_fmt", "s16",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )
        entry = {
            "index": i,
            "file": str(out_path),
            "start_seconds": round(start, 3),
            "duration_seconds": round(duration, 3),
        }
        results.append(entry)
        print(
            f"  chunk_{i:03d}.wav  start={start:.2f}s  dur={duration:.2f}s",
            flush=True,
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="VAD-based audio chunking via ffmpeg silencedetect")
    parser.add_argument("--input",            required=True,  help="Input WAV file (16kHz mono)")
    parser.add_argument("--output_dir",       required=True,  help="Directory for output chunks")
    parser.add_argument("--ffmpeg",           required=True,  help="Path to ffmpeg.exe")
    parser.add_argument("--max_seconds",      type=float, default=28.0,  help="Maximum chunk duration (seconds)")
    parser.add_argument("--silence_noise",    default="-30dB",           help="Noise floor for silencedetect")
    parser.add_argument("--silence_duration", type=float, default=0.3,   help="Minimum silence duration (seconds)")
    args = parser.parse_args()

    print(f"Input:          {args.input}", flush=True)
    print(f"Max chunk:      {args.max_seconds}s", flush=True)
    print(f"Silence noise:  {args.silence_noise}", flush=True)
    print(f"Silence min dur:{args.silence_duration}s", flush=True)

    total_duration = get_duration(args.ffmpeg, args.input)
    print(f"Total duration: {total_duration:.2f}s", flush=True)

    silences = detect_silences(
        args.ffmpeg, args.input, args.silence_noise, args.silence_duration
    )
    print(f"Silence regions: {len(silences)}", flush=True)

    split_points = compute_split_points(silences, total_duration, args.max_seconds)
    n_chunks = len(split_points) - 1
    print(f"Chunks planned: {n_chunks}", flush=True)

    output_dir = Path(args.output_dir)
    chunk_results = extract_chunks(args.ffmpeg, args.input, split_points, output_dir)

    meta = {
        "input": args.input,
        "total_duration_seconds": round(total_duration, 3),
        "max_chunk_seconds": args.max_seconds,
        "silence_noise": args.silence_noise,
        "silence_duration": args.silence_duration,
        "silence_count": len(silences),
        "chunk_count": len(chunk_results),
        "split_points": [round(p, 3) for p in split_points],
        "chunks": chunk_results,
    }
    meta_path = output_dir / "vad_splits.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metadata: {meta_path}", flush=True)
    print(f"Done. {len(chunk_results)} chunks created.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
