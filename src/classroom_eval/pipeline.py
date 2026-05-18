"""Core orchestration for Monte Carlo evaluation and uncertainty routing."""

from __future__ import annotations

from dataclasses import dataclass

from .entropy import cluster_distribution_summary, decide_review
from .models import EvaluationSample, ReviewDecision, TeachingSlice
from .samplers import EvaluationSampler


@dataclass(slots=True)
class ClassroomEvaluationPipeline:
    """Run repeated evaluations and route uncertain slices to experts."""

    sampler: EvaluationSampler
    samples_per_slice: int = 20
    temperature: float = 0.7
    entropy_threshold: float = 0.55
    similarity_threshold: float = 0.72

    def evaluate_slice(self, teaching_slice: TeachingSlice) -> tuple[ReviewDecision, tuple[EvaluationSample, ...], str]:
        """Evaluate one slice and return decision, raw samples, and disagreement summary."""

        if self.samples_per_slice <= 0:
            raise ValueError("samples_per_slice must be positive")

        samples = tuple(
            self.sampler.sample(teaching_slice, temperature=self.temperature)
            for _ in range(self.samples_per_slice)
        )
        decision = decide_review(
            teaching_slice.slice_id,
            samples,
            threshold=self.entropy_threshold,
            similarity_threshold=self.similarity_threshold,
        )
        disagreement_summary = cluster_distribution_summary(decision.clusters)
        return decision, samples, disagreement_summary
