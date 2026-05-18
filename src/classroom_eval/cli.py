"""Command-line demo for the classroom evaluation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import TranscriptSegment
from .pipeline import ClassroomEvaluationPipeline
from .samplers import RuleBasedClassroomSampler
from .slicing import slice_transcript


def _load_segments(path: Path) -> list[TranscriptSegment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [TranscriptSegment(**item) for item in payload["segments"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic-entropy classroom evaluation demo.")
    parser.add_argument("transcript", type=Path, help="Path to WhisperX-style transcript JSON")
    parser.add_argument("--samples", type=int, default=20, help="Monte Carlo samples per slice")
    parser.add_argument("--threshold", type=float, default=0.55, help="Semantic entropy review threshold")
    args = parser.parse_args()

    segments = _load_segments(args.transcript)
    slices = slice_transcript(segments, max_duration_seconds=15 * 60, activity_markers={"小组讨论", "课堂总结"})
    pipeline = ClassroomEvaluationPipeline(
        sampler=RuleBasedClassroomSampler(seed=42),
        samples_per_slice=args.samples,
        entropy_threshold=args.threshold,
    )

    output = []
    for teaching_slice in slices:
        decision, _samples, summary = pipeline.evaluate_slice(teaching_slice)
        output.append(
            {
                "slice_id": decision.slice_id,
                "start": teaching_slice.start,
                "end": teaching_slice.end,
                "entropy": round(decision.entropy, 4),
                "threshold": decision.threshold,
                "risk_tag": decision.risk_tag,
                "majority_label": decision.majority_label,
                "majority_score": round(decision.majority_score, 2),
                "disagreement_summary": summary,
            }
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
