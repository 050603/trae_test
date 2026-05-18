"""A lightweight active-learning case store for boundary examples."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .entropy import _tokenize


@dataclass(frozen=True, slots=True)
class ExpertCase:
    """One expert-adjudicated classroom boundary case."""

    case_id: str
    transcript: str
    expert_score: float
    expert_rationale: str
    model_error_reason: str


class JsonlCaseStore:
    """Persist expert cases as JSONL and retrieve similar few-shot examples."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, case: ExpertCase) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")

    def all_cases(self) -> list[ExpertCase]:
        if not self.path.exists():
            return []
        cases = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    cases.append(ExpertCase(**json.loads(line)))
        return cases

    def retrieve(self, query: str, *, top_k: int = 3) -> list[ExpertCase]:
        scored = [(self._cosine(query, case.transcript), case) for case in self.all_cases()]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [case for score, case in scored[:top_k] if score > 0]

    @staticmethod
    def _cosine(left: str, right: str) -> float:
        left_tokens = _tokenize(left)
        right_tokens = _tokenize(right)
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = len(left_tokens & right_tokens)
        return overlap / math.sqrt(len(left_tokens) * len(right_tokens))
