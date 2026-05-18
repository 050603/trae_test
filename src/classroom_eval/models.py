"""Typed data models for the classroom evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """A speaker-attributed transcript segment with precise timestamps."""

    start: float
    end: float
    text: str
    speaker: str

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("segment end time must be greater than or equal to start time")
        if not self.text.strip():
            raise ValueError("segment text cannot be empty")
        if not self.speaker.strip():
            raise ValueError("speaker cannot be empty")


@dataclass(frozen=True, slots=True)
class TeachingSlice:
    """A 10-15 minute or activity-based classroom observation slice."""

    slice_id: str
    start: float
    end: float
    segments: tuple[TranscriptSegment, ...]
    activity: str | None = None

    @property
    def transcript_text(self) -> str:
        """Return a compact speaker-prefixed transcript for prompt construction."""

        return "\n".join(f"{s.speaker} [{s.start:.3f}-{s.end:.3f}]: {s.text}" for s in self.segments)


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    """One stochastic LLM evaluation sample for a teaching slice."""

    label: str
    score: float
    rationale: str
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 7:
            raise ValueError("CLASS-style score must be in the inclusive range [0, 7]")
        if not self.label.strip():
            raise ValueError("label cannot be empty")
        if not self.rationale.strip():
            raise ValueError("rationale cannot be empty")


@dataclass(frozen=True, slots=True)
class SemanticCluster:
    """A cluster of semantically equivalent stochastic evaluation samples."""

    cluster_id: str
    samples: tuple[EvaluationSample, ...]
    probability: float
    representative: EvaluationSample


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    """Routing decision produced by semantic-entropy uncertainty gating."""

    slice_id: str
    entropy: float
    threshold: float
    needs_human_review: bool
    majority_label: str
    majority_score: float
    clusters: tuple[SemanticCluster, ...]

    @property
    def risk_tag(self) -> str:
        """Human-readable queue tag for front-end workbenches."""

        return "高危/不确定" if self.needs_human_review else "自动采纳"
