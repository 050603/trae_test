from classroom_eval.active_learning import ExpertCase, JsonlCaseStore


def test_case_store_retrieves_similar_cases(tmp_path):
    store = JsonlCaseStore(tmp_path / "cases.jsonl")
    store.add(
        ExpertCase(
            case_id="case-1",
            transcript="教师 为什么 请解释 证据",
            expert_score=6.0,
            expert_rationale="开放式追问促进解释。",
            model_error_reason="低估了追问价值",
        )
    )
    store.add(
        ExpertCase(
            case_id="case-2",
            transcript="教师 批评 太慢 不对",
            expert_score=2.0,
            expert_rationale="负向控制语言较多。",
            model_error_reason="忽略情绪压力",
        )
    )
    results = store.retrieve("为什么 请解释 实验证据", top_k=1)
    assert results[0].case_id == "case-1"
