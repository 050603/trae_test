"""Evaluation samplers used by the classroom evaluation pipeline."""

from __future__ import annotations

import random
from typing import Protocol

from .models import EvaluationSample, TeachingSlice


class EvaluationSampler(Protocol):
    """Protocol for stochastic LLM evaluators."""

    def sample(self, teaching_slice: TeachingSlice, *, temperature: float) -> EvaluationSample:
        """Return one stochastic evaluation sample."""


class RuleBasedClassroomSampler:
    """A deterministic local stand-in for GPT/Llama Monte Carlo sampling.

    It introduces controlled variation so the semantic-entropy pipeline can be
    tested without network calls. Replace this class with an OpenAI, vLLM, or
    HuggingFace-backed sampler in production.
    """

    positive_terms = ("为什么", "你怎么想", "请解释", "很好", "支持", "尝试", "启发")
    negative_terms = ("不对", "太慢", "闭嘴", "必须", "批评", "压迫")

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def sample(self, teaching_slice: TeachingSlice, *, temperature: float = 0.7) -> EvaluationSample:
        text = teaching_slice.transcript_text
        positive = sum(text.count(term) for term in self.positive_terms)
        negative = sum(text.count(term) for term in self.negative_terms)
        jitter = self._rng.uniform(-temperature, temperature)
        score = min(7.0, max(1.0, 4.0 + positive * 0.7 - negative * 0.9 + jitter))

        if score >= 5.2:
            label = "教学支持较强"
            rationale = "教师使用开放式提问或积极反馈，促进学生解释和表达。"
        elif score <= 3.2:
            label = "情感氛围存在风险"
            rationale = "教师语言中出现压迫或否定信号，需要专家核查课堂氛围。"
        else:
            label = self._rng.choice(["教学支持中等", "情感氛围中性"])
            rationale = self._rng.choice(
                [
                    "片段既包含推进课堂的指令，也缺少充分追问证据。",
                    "模型对教师反馈的支持性和控制性感到分歧。",
                ]
            )

        return EvaluationSample(label=label, score=round(score, 2), rationale=rationale)
