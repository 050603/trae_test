from classroom_eval.models import TranscriptSegment
from classroom_eval.slicing import slice_transcript


def test_slice_transcript_splits_on_duration():
    segments = [
        TranscriptSegment(0, 10, "开场", "SPEAKER_00"),
        TranscriptSegment(11, 20, "继续", "SPEAKER_00"),
        TranscriptSegment(40, 50, "下一段", "SPEAKER_00"),
    ]
    slices = slice_transcript(segments, max_duration_seconds=30)
    assert [s.slice_id for s in slices] == ["slice-001", "slice-002"]
    assert slices[0].start == 0
    assert slices[1].start == 40


def test_slice_transcript_splits_on_activity_marker():
    segments = [
        TranscriptSegment(0, 5, "讲解概念", "SPEAKER_00"),
        TranscriptSegment(6, 8, "现在小组讨论", "SPEAKER_00"),
    ]
    slices = slice_transcript(segments, activity_markers={"小组讨论"})
    assert len(slices) == 2
    assert slices[1].segments[0].text == "现在小组讨论"
