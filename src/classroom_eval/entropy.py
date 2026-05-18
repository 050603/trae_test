"""Semantic clustering and entropy-based review routing.

The implementation intentionally keeps dependencies light so the uncertainty
pipeline can be validated in notebooks before wiring in DeBERTa, pyannote, or a
production vector database. Replace ``text_similarity`` with an embedding cosine
similarity function when a stronger model is available.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable

from .models import EvaluationSample, ReviewDecision, SemanticCluster

_TOKEN_RE = re.compile(r"[\w]+|[\u4e00-\u9fff]", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def text_similarity(left: str, right: str) -> float:
    """Return a deterministic lexical similarity score in ``[0, 1]``.

    This Jaccard baseline is deliberately transparent for early calibration; in
    production it should be swapped for an embedding model such as DeBERTa or a
    domain-tuned sentence transformer.
    """

    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _sample_signature(sample: EvaluationSample) -> str:
    return f"{sample.label} score={round(sample.score)} {sample.rationale}"


def cluster_samples(
    samples: Iterable[EvaluationSample],
    *,
    similarity_threshold: float = 0.72,
    score_tolerance: float = 1.0,
) -> tuple[SemanticCluster, ...]:
    """Group semantically equivalent LLM evaluation samples.

    Samples are placed into the first compatible cluster when they share the
    same label, have close CLASS-style scores, and pass the text similarity
    threshold against the cluster representative.
    """

    sample_list = list(samples)
    if not sample_list:
        raise ValueError("at least one evaluation sample is required")

    buckets: list[list[EvaluationSample]] = []
    for sample in sample_list:
        for bucket in buckets:
            representative = bucket[0]
            same_label = representative.label == sample.label
            close_score = abs(representative.score - sample.score) <= score_tolerance
            similar_text = text_similarity(_sample_signature(representative), _sample_signature(sample)) >= similarity_threshold
            if same_label and close_score and similar_text:
                bucket.append(sample)
                break
        else:
            buckets.append([sample])

    total = len(sample_list)
    clusters = []
    for index, bucket in enumerate(sorted(buckets, key=len, reverse=True), start=1):
        clusters.append(
            SemanticCluster(
                cluster_id=f"C{index}",
                samples=tuple(bucket),
                probability=len(bucket) / total,
                representative=bucket[0],
            )
        )
    return tuple(clusters)


def semantic_entropy(clusters: Iterable[SemanticCluster]) -> float:
    """Compute Shannon entropy over semantic cluster probabilities."""

    entropy = 0.0
    for cluster in clusters:
        if cluster.probability <= 0:
            continue
        entropy -= cluster.probability * math.log(cluster.probability)
    return entropy


def _majority_vote(samples: Iterable[EvaluationSample]) -> tuple[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample.score)
    label, scores = max(grouped.items(), key=lambda item: (len(item[1]), sum(item[1]) / len(item[1])))
    return label, sum(scores) / len(scores)


def cluster_distribution_summary(clusters: Iterable[SemanticCluster]) -> str:
    """Summarize cluster disagreement for an expert workbench."""

    parts = []
    for cluster in clusters:
        percent = round(cluster.probability * 100)
        parts.append(f"{percent}% 样本认为：{cluster.representative.label}（约 {cluster.representative.score:.1f} 分）")
    return "；".join(parts)


def decide_review(
    slice_id: str,
    samples: Iterable[EvaluationSample],
    *,
    threshold: float,
    similarity_threshold: float = 0.72,
) -> ReviewDecision:
    """Route a slice to auto-accept or human review using semantic entropy."""

    sample_list = list(samples)
    clusters = cluster_samples(sample_list, similarity_threshold=similarity_threshold)
    entropy = semantic_entropy(clusters)
    majority_label, majority_score = _majority_vote(sample_list)
    return ReviewDecision(
        slice_id=slice_id,
        entropy=entropy,
        threshold=threshold,
        needs_human_review=entropy > threshold,
        majority_label=majority_label,
        majority_score=majority_score,
        clusters=clusters,
    )


def label_counts(samples: Iterable[EvaluationSample]) -> Counter[str]:
    """Return label counts for diagnostics and calibration notebooks."""

    return Counter(sample.label for sample in samples)
