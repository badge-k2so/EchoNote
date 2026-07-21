from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from otoweave_app.audio import AdaptiveVad, SAMPLE_RATE, convert_audio_to_pcm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ReazonSpeech K2 and Japanese Parakeet CTC.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parakeet-dir", type=Path, required=True)
    parser.add_argument("--school-audio", type=Path)
    parser.add_argument("--school-seconds", type=float, default=600.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--worker-model", choices=("reazon_k2", "parakeet_ctc"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--result", type=Path)
    return parser.parse_args()


def normalize_ja(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if ch.isalnum() or "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff")


def edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    reference = normalize_ja(reference)
    hypothesis = normalize_ja(hypothesis)
    return edit_distance(reference, hypothesis) / max(1, len(reference))


def write_chunks(case_dir: Path, pcm_path: Path, limit_seconds: float | None) -> list[dict[str, object]]:
    case_dir.mkdir(parents=True, exist_ok=True)
    vad = AdaptiveVad(max_chunk_seconds=12.0)
    chunks: list[dict[str, object]] = []
    processed_samples = 0
    sample_limit = int(limit_seconds * SAMPLE_RATE) if limit_seconds else None
    with pcm_path.open("rb") as stream:
        while sample_limit is None or processed_samples < sample_limit:
            remaining = 1600 if sample_limit is None else min(1600, sample_limit - processed_samples)
            if remaining <= 0:
                break
            data = stream.read(remaining * 2)
            if not data:
                break
            samples = np.frombuffer(data, dtype=np.int16).copy()
            for chunk in vad.process(samples, processed_samples / SAMPLE_RATE):
                path = case_dir / f"chunk_{len(chunks):04d}.npy"
                np.save(path, chunk.samples)
                chunks.append({"path": str(path), "start": chunk.start, "end": chunk.end})
            processed_samples += samples.size
        for chunk in vad.flush():
            path = case_dir / f"chunk_{len(chunks):04d}.npy"
            np.save(path, chunk.samples)
            chunks.append({"path": str(path), "start": chunk.start, "end": chunk.end})
    return chunks


def load_references(model_dir: Path) -> dict[str, str]:
    references: dict[str, str] = {}
    path = model_dir / "test_wavs" / "transcripts.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        if ", " in line:
            name, text = line.split(", ", 1)
            references[name] = text
    return references


def prepare_manifest(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    chunks_root = output_dir / "chunks"
    chunks_root.mkdir(parents=True, exist_ok=True)
    ffmpeg = PROJECT_ROOT / "engines" / "ffmpeg" / "ffmpeg.exe"
    references = load_references(args.parakeet_dir)
    cases: list[dict[str, object]] = []

    for name in ("test_ja_1", "test_ja_2"):
        source = args.parakeet_dir / "test_wavs" / f"{name}.wav"
        pcm = output_dir / f"{name}.pcm"
        convert_audio_to_pcm(ffmpeg, source, pcm)
        samples = np.fromfile(pcm, dtype=np.int16)
        direct_dir = chunks_root / f"{name}_direct"
        direct_dir.mkdir(parents=True, exist_ok=True)
        direct_path = direct_dir / "chunk_0000.npy"
        np.save(direct_path, samples)
        direct_chunks = [{"path": str(direct_path), "start": 0.0, "end": samples.size / SAMPLE_RATE}]
        cases.append(
            {
                "name": f"{name}_direct",
                "source": str(source),
                "reference": references[name],
                "chunks": direct_chunks,
            }
        )
        chunks = write_chunks(chunks_root / f"{name}_vad", pcm, None)
        pcm.unlink(missing_ok=True)
        cases.append(
            {
                "name": f"{name}_vad",
                "source": str(source),
                "reference": references[name],
                "chunks": chunks,
            }
        )

    if args.school_audio:
        source = args.school_audio.resolve()
        pcm = output_dir / "school_audio.pcm"
        convert_audio_to_pcm(ffmpeg, source, pcm)
        chunks = write_chunks(chunks_root / "school_audio", pcm, args.school_seconds)
        pcm.unlink(missing_ok=True)
        cases.append({"name": "school_audio", "source": str(source), "reference": "", "chunks": chunks})

    manifest = output_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "threads": max(1, args.threads),
                "parakeet_dir": str(args.parakeet_dir.resolve()),
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


class ParakeetRecognizer:
    def __init__(self, model_dir: Path, num_threads: int) -> None:
        import sherpa_onnx

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(model_dir / "model.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu",
        )

    def transcribe(self, samples: np.ndarray) -> str:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples.astype(np.float32) / 32768.0)
        self.recognizer.decode_stream(stream)
        return str(stream.result.text).strip()


def run_worker(args: argparse.Namespace) -> int:
    from otoweave_app.asr import JapaneseRecognizer

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    threads = int(manifest["threads"])
    load_started = time.perf_counter()
    if args.worker_model == "reazon_k2":
        recognizer = JapaneseRecognizer(num_threads=threads)
    else:
        recognizer = ParakeetRecognizer(Path(manifest["parakeet_dir"]), threads)
    load_seconds = time.perf_counter() - load_started

    cases = []
    total_audio_seconds = 0.0
    total_asr_seconds = 0.0
    for case in manifest["cases"]:
        texts = []
        chunk_results = []
        case_audio_seconds = 0.0
        case_started = time.perf_counter()
        for index, chunk in enumerate(case["chunks"]):
            samples = np.load(chunk["path"])
            text = recognizer.transcribe(samples)
            texts.append(text)
            normalized = samples.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(normalized * normalized) + 1e-12))
            chunk_results.append(
                {
                    "index": index,
                    "start": round(float(chunk["start"]), 3),
                    "end": round(float(chunk["end"]), 3),
                    "rms": round(rms, 6),
                    "chars": len(text),
                    "text": text,
                }
            )
            case_audio_seconds += float(chunk["end"]) - float(chunk["start"])
        case_asr_seconds = time.perf_counter() - case_started
        transcript = "\n".join(text for text in texts if text)
        total_audio_seconds += case_audio_seconds
        total_asr_seconds += case_asr_seconds
        item = {
            "name": case["name"],
            "audio_seconds": round(case_audio_seconds, 3),
            "asr_seconds": round(case_asr_seconds, 3),
            "rtf": round(case_asr_seconds / max(0.001, case_audio_seconds), 4),
            "transcript": transcript,
            "chunk_results": chunk_results,
        }
        if case["reference"]:
            item["reference"] = case["reference"]
            item["cer"] = round(cer(case["reference"], transcript), 4)
        cases.append(item)

    args.result.write_text(
        json.dumps(
            {
                "model": args.worker_model,
                "threads": threads,
                "model_load_seconds": round(load_seconds, 3),
                "audio_seconds": round(total_audio_seconds, 3),
                "asr_seconds": round(total_asr_seconds, 3),
                "rtf": round(total_asr_seconds / max(0.001, total_audio_seconds), 4),
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def run_model_process(model: str, manifest: Path, output_dir: Path) -> dict[str, object]:
    import psutil

    result_path = output_dir / f"{model}.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output-dir",
        str(output_dir),
        "--parakeet-dir",
        str(json.loads(manifest.read_text(encoding="utf-8"))["parakeet_dir"]),
        "--worker-model",
        model,
        "--manifest",
        str(manifest),
        "--result",
        str(result_path),
    ]
    started = time.perf_counter()
    process = subprocess.Popen(command)
    measured = psutil.Process(process.pid)
    peak_rss = 0
    while process.poll() is None:
        try:
            processes = [measured, *measured.children(recursive=True)]
            current_rss = sum(child.memory_info().rss for child in processes if child.is_running())
            peak_rss = max(peak_rss, current_rss)
        except psutil.Error:
            pass
        time.sleep(0.05)
    if process.returncode != 0:
        raise RuntimeError(f"{model} benchmark failed with exit code {process.returncode}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["peak_rss_mb"] = round(peak_rss / (1024 * 1024), 1)
    result["process_wall_seconds"] = round(time.perf_counter() - started, 3)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def school_terms(text: str) -> dict[str, bool]:
    terms = [
        "ディスレクシア",
        "読み書き",
        "学習障害",
        "発達障害",
        "脳",
        "研究",
        "支援",
        "学習",
        "評価",
        "専門家",
    ]
    normalized = normalize_ja(text)
    return {term: normalize_ja(term) in normalized for term in terms}


def write_report(output_dir: Path, results: list[dict[str, object]], school_seconds: float) -> Path:
    by_model = {str(result["model"]): result for result in results}
    school_limit = "full audio" if school_seconds <= 0 else f"{school_seconds:.0f} seconds"
    lines = [
        "# Japanese ASR CPU comparison",
        "",
        f"- Threads: {results[0]['threads']}",
        f"- School audio limit: {school_limit}",
        "",
        "## Performance",
        "",
        "| Model | Load | ASR | Audio | RTF | Peak RSS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("reazon_k2", "parakeet_ctc"):
        result = by_model[model]
        lines.append(
            f"| {model} | {result['model_load_seconds']:.3f}s | {result['asr_seconds']:.3f}s | "
            f"{result['audio_seconds']:.3f}s | {result['rtf']:.4f} | {result['peak_rss_mb']:.1f}MB |"
        )

    lines.extend(
        [
            "",
            "## Reference CER",
            "",
            "| Case | Reazon K2 | Parakeet CTC |",
            "|---|---:|---:|",
        ]
    )
    for case_name in ("test_ja_1_direct", "test_ja_1_vad", "test_ja_2_direct", "test_ja_2_vad"):
        values = {}
        for model, result in by_model.items():
            values[model] = next(case for case in result["cases"] if case["name"] == case_name)
        lines.append(f"| {case_name} | {values['reazon_k2']['cer']:.2%} | {values['parakeet_ctc']['cer']:.2%} |")

    lines.extend(["", "## School audio", ""])
    for model in ("reazon_k2", "parakeet_ctc"):
        result = by_model[model]
        case = next((case for case in result["cases"] if case["name"] == "school_audio"), None)
        if case is None:
            continue
        lines.extend(
            [
                f"### {model}",
                "",
                f"- RTF: {case['rtf']:.4f}",
                f"- Terms: `{json.dumps(school_terms(case['transcript']), ensure_ascii=False)}`",
                "",
                case["transcript"],
                "",
            ]
        )
    report = output_dir / "comparison.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    if args.worker_model:
        if not args.manifest or not args.result:
            raise ValueError("--manifest and --result are required in worker mode")
        return run_worker(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = prepare_manifest(args)
    results = [
        run_model_process("reazon_k2", manifest, args.output_dir),
        run_model_process("parakeet_ctc", manifest, args.output_dir),
    ]
    report = write_report(args.output_dir, results, args.school_seconds)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
