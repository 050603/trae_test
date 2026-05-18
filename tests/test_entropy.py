from classroom_eval.entropy import cluster_samples, decide_review, semantic_entropy
from classroom_eval.models import EvaluationSample


def test_semantic_entropy_is_low_for_consensus_samples():
    samples = [
        EvaluationSample("教学支持较强", 6.0, "教师开放式提问并积极反馈。"),
        EvaluationSample("教学支持较强", 6.2, "教师开放式提问并积极反馈学生表达。"),
        EvaluationSample("教学支持较强", 5.8, "教师积极反馈并促进学生解释。"),
    ]
    clusters = cluster_samples(samples, similarity_threshold=0.3)
    assert len(clusters) == 1
    assert semantic_entropy(clusters) == 0


def test_decide_review_flags_high_entropy_disagreement():
    samples = [
        EvaluationSample("教学支持较强", 6.0, "教师启发学生解释。"),
        EvaluationSample("情感氛围存在风险", 2.0, "教师语言具有压迫感。"),
        EvaluationSample("教学支持中等", 4.0, "证据不足，支持性一般。"),
    ]
    decision = decide_review("slice-001", samples, threshold=0.5)
    assert decision.needs_human_review is True
    assert decision.risk_tag == "高危/不确定"
