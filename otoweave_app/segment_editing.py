"""Pure transcript-editing helpers shared by the controller.

These functions operate on TranscriptSegment lists without touching
storage, locks, or the event queue, so they can be unit-tested directly.
"""
from __future__ import annotations

import re
from datetime import datetime

from .models import TranscriptSegment


def mark_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def next_segment_id(segments: list[TranscriptSegment]) -> str:
    numbers = []
    for segment in segments:
        match = re.fullmatch(r"seg_(\d+)", segment.id)
        if match:
            numbers.append(int(match.group(1)))
    return f"seg_{max(numbers, default=0) + 1:04d}"


def remap_edited_blocks(
    segments: list[TranscriptSegment],
    blocks: list[str],
) -> list[TranscriptSegment]:
    """Place edited paragraphs on the old timeline without losing marks."""
    timeline_start = segments[0].start
    timeline_end = max(timeline_start, segments[-1].end)
    duration = timeline_end - timeline_start
    weights = [max(1, len(block.replace(" ", ""))) for block in blocks]
    total_weight = sum(weights)

    spans: list[tuple[float, float]] = []
    consumed = 0
    for index, weight in enumerate(weights):
        start = timeline_start + duration * consumed / total_weight
        consumed += weight
        end = (
            timeline_end
            if index == len(weights) - 1
            else timeline_start + duration * consumed / total_weight
        )
        spans.append((start, end))

    sources_by_block: list[list[TranscriptSegment]] = [[] for _ in blocks]
    for source_index, segment in enumerate(segments):
        if duration <= 0:
            block_index = min(
                len(blocks) - 1,
                source_index * len(blocks) // len(segments),
            )
        else:
            midpoint = (segment.start + segment.end) / 2
            block_index = len(blocks) - 1
            for candidate, (_, end) in enumerate(spans[:-1]):
                if midpoint < end:
                    block_index = candidate
                    break
        sources_by_block[block_index].append(segment)

    remapped: list[TranscriptSegment] = []
    used_ids: set[str] = set()
    for index, (block, (start, end), sources) in enumerate(
        zip(blocks, spans, sources_by_block)
    ):
        primary = (
            sources[0]
            if sources
            else segments[min(len(segments) - 1, index * len(segments) // len(blocks))]
        )
        segment_id = primary.id
        if segment_id in used_ids:
            segment_id = f"{segments[0].id}_edit_{index + 1:04d}"
        while segment_id in used_ids:
            segment_id += "_"
        used_ids.add(segment_id)

        speakers = {segment.speaker for segment in sources}
        speaker = (
            primary.speaker
            if not sources
            else (next(iter(speakers)) if len(speakers) == 1 else "")
        )
        remapped.append(
            TranscriptSegment(
                id=segment_id,
                start=start,
                end=end,
                text=block,
                speaker=speaker,
                status=primary.status,
                important=any(segment.important for segment in sources),
                unclear=any(segment.unclear for segment in sources),
                question=any(segment.question for segment in sources),
                important_at=next(
                    (segment.important_at for segment in sources if segment.important),
                    "",
                ),
                unclear_at=next(
                    (segment.unclear_at for segment in sources if segment.unclear),
                    "",
                ),
                question_at=next(
                    (segment.question_at for segment in sources if segment.question),
                    "",
                ),
                edited=True,
            )
        )
    return remapped


def rename_speaker(segments: list[TranscriptSegment], old: str, new: str) -> int:
    """Rename every segment whose speaker matches `old` to `new` in place.

    Returns the number of segments changed. Segments already at `new`
    (including the no-op `old == new`) are left untouched and not counted,
    so callers can safely call this once per (old, new) pair without
    double-marking already-renamed segments as edited.
    """
    changed = 0
    for segment in segments:
        if segment.speaker == old and segment.speaker != new:
            segment.speaker = new
            segment.edited = True
            changed += 1
    return changed


def transfer_marks(
    previous_segments: list[TranscriptSegment],
    new_segments: list[TranscriptSegment],
) -> None:
    """Carry ★/?/! marks from an old transcript to the nearest new segments."""
    if not new_segments:
        return
    for previous in previous_segments:
        if not previous.important and not previous.unclear and not previous.question:
            continue
        center = (previous.start + previous.end) / 2
        nearest = min(
            new_segments,
            key=lambda segment: min(abs(center - segment.start), abs(center - segment.end)),
        )
        if previous.important:
            nearest.important = True
            nearest.important_at = previous.important_at
        if previous.unclear:
            nearest.unclear = True
            nearest.unclear_at = previous.unclear_at
        if previous.question:
            nearest.question = True
            nearest.question_at = previous.question_at
