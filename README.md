# 人机协同课堂评价系统（起步版）

本仓库提供一个可在 Python 3.11 中直接运行的课堂评价核心算法底座，用于先验证“多次采样 → 语义聚类 → 语义熵 → 自动拦截/专家复核”的闭环，再逐步接入 WhisperX、真实大模型、前端专家工作台与向量数据库。

## 目标架构

1. **多模态预处理与转录**：生产环境建议使用 WhisperX + `pyannote.audio`，输出包含毫秒级 `start`、`end`、`speaker`、`text` 的 JSON。
2. **微观教学切片**：将结构化转录稿按 10-15 分钟窗口或“讲解 / 小组讨论 / 总结”等活动边界切分。
3. **大模型初评**：对每个切片执行 CLASS 量表、布鲁姆分类学等教育指标评价。
4. **不确定性拦截**：对同一切片执行 Monte Carlo 多次采样，聚类语义等价回答，计算语义熵：

   ```text
   SE(x) = - Σ P(C_i|x) log P(C_i|x)
   ```

   熵低于阈值时自动采纳多数票；熵高于阈值时进入“高危/不确定”专家队列。
5. **主动学习**：专家裁定后的边界案例写入案例库，后续作为 few-shot / CoT 示例检索进提示词。

## 当前实现

- `classroom_eval.slicing.slice_transcript`：按时间窗口和活动标记生成教学切片。
- `classroom_eval.samplers.RuleBasedClassroomSampler`：本地规则采样器，用于无网络验证 Monte Carlo 管道；生产环境可替换为 GPT、Llama 或其他模型调用。
- `classroom_eval.entropy`：语义聚类、Shannon 熵计算、多数票与人工审核路由。
- `classroom_eval.active_learning.JsonlCaseStore`：轻量 JSONL 专家案例库，模拟向量数据库检索相似边界案例。
- `classroom-eval` CLI：读取 WhisperX 风格 JSON 并输出每个切片的审核决策。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
classroom-eval examples/sample_transcript.json --samples 20 --threshold 0.55
```

示例输出会包含：

- `entropy`：语义熵。
- `risk_tag`：`自动采纳` 或 `高危/不确定`。
- `majority_label` / `majority_score`：低熵情况下可写入最终数据库的多数票结果。
- `disagreement_summary`：给专家工作台展示的中文分歧摘要。

## 接入 WhisperX 的建议

本仓库不直接下载或调用需要授权的 HuggingFace 模型。落地时请：

1. 在 HuggingFace 为 `pyannote/speaker-diarization-3.1` 申请访问授权。
2. 将访问令牌放入环境变量，例如 `HUGGINGFACE_TOKEN`，不要提交到 Git。
3. 使用 WhisperX 输出与 `examples/sample_transcript.json` 相同字段的数据结构。
4. 将真实 JSON 传入 `classroom-eval` 或 Notebook 中的 `slice_transcript`。

## 生产化替换点

- 将 `RuleBasedClassroomSampler` 替换为真实 LLM sampler，并设置 `temperature=0.7`、`samples_per_slice=20`。
- 将 `entropy.text_similarity` 替换为 DeBERTa / Sentence Transformer embedding cosine 相似度。
- 用人工标注历史课程校准 `entropy_threshold`，不要直接沿用示例阈值。
- 将 `JsonlCaseStore` 替换为 Milvus、pgvector、Qdrant 等向量数据库。
- 在专家前端中根据 `TeachingSlice.start/end` 自动截取 1-3 分钟争议视频片段，并展示 `disagreement_summary`。
