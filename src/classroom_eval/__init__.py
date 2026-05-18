"""Human-in-the-loop classroom evaluation utilities."""

from .entropy import cluster_samples, decide_review, semantic_entropy
from .models import (
    EvaluationSample,
    ReviewDecision,
    SemanticCluster,
    TeachingSlice,
    TranscriptSegment,
)
from .pipeline import ClassroomEvaluationPipeline
from .slicing import slice_transcript

__all__ = [
    "ClassroomEvaluationPipeline",
    "EvaluationSample",
    "ReviewDecision",
    "SemanticCluster",
    "TeachingSlice",
    "TranscriptSegment",
    "cluster_samples",
    "decide_review",
    "semantic_entropy",
    "slice_transcript",
]
