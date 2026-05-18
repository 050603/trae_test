"""Utilities for converting timestamped transcripts into teaching slices."""

from __future__ import annotations

from collections.abc import Iterable

from .models import TeachingSlice, TranscriptSegment


def slice_transcript(
    segments: Iterable[TranscriptSegment],
    *,
    max_duration_seconds: float = 15 * 60,
    activity_markers: set[str] | None = None,
) -> tuple[TeachingSlice, ...]:
    """Split transcript segments into time-bounded or activity-bounded slices.

    ``activity_markers`` can contain teacher utterances such as ``小组讨论`` or
    ``课堂总结``. When a marker appears after the current slice has content, a new
    slice starts at that segment so downstream prompts align with teaching
    activities instead of arbitrary token windows.
    """

    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    if not ordered:
        return tuple()

    markers = activity_markers or set()
    slices: list[TeachingSlice] = []
    current: list[TranscriptSegment] = []
    current_start = ordered[0].start

    def flush(activity: str | None = None) -> None:
        nonlocal current, current_start
        if not current:
            return
        slice_number = len(slices) + 1
        slices.append(
            TeachingSlice(
                slice_id=f"slice-{slice_number:03d}",
                start=current[0].start,
                end=current[-1].end,
                segments=tuple(current),
                activity=activity,
            )
        )
        current = []

    for segment in ordered:
        contains_marker = next((marker for marker in markers if marker in segment.text), None)
        would_exceed = current and (segment.end - current_start > max_duration_seconds)
        if would_exceed or (contains_marker and current):
            flush(activity=contains_marker)
            current_start = segment.start
        current.append(segment)

    flush()
    return tuple(slices)
