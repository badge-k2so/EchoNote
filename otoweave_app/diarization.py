"""Offline speaker diarization: sherpa-onnx (pyannote segmentation + 3D-Speaker
embedding + fast clustering) applied to a finished transcript.

This mirrors scripts/prototypes/diarization_prototype.py, which validated the approach
(RTF ~0.174 at 4 threads, ~580MB peak memory for a 71-minute file). Two
findings from that validation drive the API here:

  * Automatic speaker-count estimation (FastClusteringConfig with only a
    distance threshold) is unreliable and must never be used. num_speakers
    is always required and passed straight through to num_clusters.
  * Segment-to-speaker assignment must use overlap-time argmax (the total
    duration each diarized speaker span overlaps a transcript segment),
    never a midpoint/nearest-span heuristic. A segment is only labeled when
    the winning speaker's share of the overlapped time is high enough
    (purity_threshold) that the assignment is trustworthy; otherwise it is
    left unassigned rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE
from .models import TranscriptSegment


SEGMENTATION_MODEL = Path("models/diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx")
EMBEDDING_MODEL = Path("models/diarization/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")

# FastClusteringConfig requires a threshold even when num_clusters fixes the
# speaker count explicitly (in which case sherpa-onnx ignores the distance
# threshold entirely). Kept identical to the validated prototype for parity.
_CLUSTERING_THRESHOLD = 0.5
_MIN_DURATION_ON = 0.3
_MIN_DURATION_OFF = 0.5


def diarization_available(project_root: Path) -> bool:
    """Whether both diarization model files are present on disk.

    Mirrors the size-sanity style of asr.qwen3_asr_17_available: a present
    but truncated/corrupt download should not be reported as available."""
    root = Path(project_root)
    segmentation = root / SEGMENTATION_MODEL
    embedding = root / EMBEDDING_MODEL
    return (
        segmentation.is_file()
        and segmentation.stat().st_size > 1_000_000
        and embedding.is_file()
        and embedding.stat().st_size > 10_000_000
    )


@dataclass(frozen=True)
class DiarizedSpan:
    start: float
    end: float
    speaker: int


@dataclass(frozen=True)
class DiarizationResult:
    """`[{start, end, speaker}]`-equivalent output of one diarize() call."""

    spans: list[DiarizedSpan]

    def __iter__(self):
        return iter(self.spans)

    def __len__(self) -> int:
        return len(self.spans)


class SpeakerDiarizer:
    """Offline speaker diarization via sherpa-onnx's OfflineSpeakerDiarization.

    num_speakers must always be given explicitly (see module docstring):
    automatic speaker-count estimation is disallowed."""

    def __init__(
        self,
        segmentation_model: Path,
        embedding_model: Path,
        num_threads: int = 4,
    ) -> None:
        import sherpa_onnx

        segmentation_model = Path(segmentation_model)
        embedding_model = Path(embedding_model)
        missing = [
            str(path) for path in (segmentation_model, embedding_model) if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError("Diarization model files are missing: " + ", ".join(missing))

        self.num_threads = max(1, int(num_threads))
        self._segmentation_model = segmentation_model
        self._embedding_model = embedding_model
        self._sherpa_onnx = sherpa_onnx

    def _build(self, num_speakers: int):
        sherpa_onnx = self._sherpa_onnx
        segmentation_config = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(self._segmentation_model)
            ),
            num_threads=self.num_threads,
            debug=False,
            provider="cpu",
        )
        embedding_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(self._embedding_model),
            num_threads=self.num_threads,
            debug=False,
            provider="cpu",
        )
        clustering_config = sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers,
            threshold=_CLUSTERING_THRESHOLD,
        )
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=segmentation_config,
            embedding=embedding_config,
            clustering=clustering_config,
            min_duration_on=_MIN_DURATION_ON,
            min_duration_off=_MIN_DURATION_OFF,
        )
        if not config.validate():
            raise RuntimeError("OfflineSpeakerDiarizationConfig failed validate()")
        return sherpa_onnx.OfflineSpeakerDiarization(config)

    def diarize(self, samples: np.ndarray, num_speakers: int) -> DiarizationResult:
        """Diarize 16kHz mono float32 samples into num_speakers speakers.

        num_speakers must be a positive integer; auto-detection (leaving the
        speaker count unset) is intentionally not supported here."""
        if num_speakers is None or int(num_speakers) < 1:
            raise ValueError(
                "num_speakers must be a positive integer; automatic speaker-count "
                "estimation is disallowed (see module docstring)"
            )
        waveform = np.asarray(samples, dtype=np.float32).reshape(-1)

        diarizer = self._build(int(num_speakers))
        if diarizer.sample_rate != SAMPLE_RATE:
            raise RuntimeError(
                f"Diarizer expects sample_rate={diarizer.sample_rate}, "
                f"but samples are assumed to be {SAMPLE_RATE}"
            )
        result = diarizer.process(waveform)
        spans = [
            DiarizedSpan(start=float(seg.start), end=float(seg.end), speaker=int(seg.speaker))
            for seg in result.sort_by_start_time()
        ]
        return DiarizationResult(spans=spans)


def assign_speakers(
    segments: list[TranscriptSegment],
    result: DiarizationResult,
    purity_threshold: float = 0.9,
) -> None:
    """Assign a Japanese speaker label to each segment, in place.

    For every segment, the overlap duration with each diarized speaker span
    is summed (validated approach: overlap-time argmax, never a midpoint
    match). If the dominant speaker's share of the total overlapped time
    (its "purity") is at least purity_threshold, the segment's speaker is
    set to "話者1", "話者2", ... — numbered in ascending diarized-speaker-
    index order, not order of appearance. Segments with no overlap at all,
    or whose purity falls below the threshold, are left untouched (so a
    fresh segment keeps its default empty speaker)."""
    spans = list(result)
    if not spans:
        return
    speaker_ids = sorted({span.speaker for span in spans})
    label_by_speaker_id = {
        speaker_id: f"話者{index + 1}" for index, speaker_id in enumerate(speaker_ids)
    }
    for segment in segments:
        overlap_by_speaker: dict[int, float] = {}
        for span in spans:
            overlap = min(segment.end, span.end) - max(segment.start, span.start)
            if overlap > 0:
                overlap_by_speaker[span.speaker] = overlap_by_speaker.get(span.speaker, 0.0) + overlap
        total_overlap = sum(overlap_by_speaker.values())
        if total_overlap <= 0:
            continue
        dominant_speaker, dominant_overlap = max(
            overlap_by_speaker.items(), key=lambda item: item[1]
        )
        purity = dominant_overlap / total_overlap
        if purity >= purity_threshold:
            segment.speaker = label_by_speaker_id[dominant_speaker]
