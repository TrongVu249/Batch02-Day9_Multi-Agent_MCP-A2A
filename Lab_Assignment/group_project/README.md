# Bài Tập Nhóm — RAG Evaluation Pipeline

## Sản Phẩm Đã Chọn

> ✅ **Sản phẩm 2: RAG Evaluation Pipeline** (Yêu cầu 2)
>
> Sử dụng **DeepEval** để evaluate RAG pipeline với 4 metrics chuẩn,
> so sánh A/B giữa 2 configs, và tạo báo cáo phân tích chi tiết.

---

## Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────┐
│                   RAG Evaluation Pipeline                    │
│                                                             │
│  golden_dataset.json (16 Q&A pairs)                        │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────┐                   │
│  │         Config A: Hybrid + Rerank   │                   │
│  │  Semantic Search ──┐                │                   │
│  │                    ├── RRF Merge    │                   │
│  │  BM25 Search    ──┘        │        │                   │
│  │                     Cross-Encoder   │                   │
│  │                     Reranking       │                   │
│  └─────────────────────────────────────┘                   │
│         │                │                                  │
│         │   A/B Test     │                                  │
│         │                │                                  │
│  ┌─────────────────────────────────────┐                   │
│  │         Config B: Dense-Only        │                   │
│  │  Semantic Search (cosine sim only)  │                   │
│  │  No lexical, no reranking           │                   │
│  └─────────────────────────────────────┘                   │
│         │                                                   │
│         ▼                                                   │
│  Task 10: generate_with_citation()                         │
│  (OpenRouter → gpt-4o-mini)                                │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────────────────────────┐                  │
│  │         DeepEval Judge               │                  │
│  │  • FaithfulnessMetric                │                  │
│  │  • AnswerRelevancyMetric             │                  │
│  │  • ContextualRecallMetric            │                  │
│  │  • ContextualPrecisionMetric         │                  │
│  └──────────────────────────────────────┘                  │
│         │                                                   │
│         ▼                                                   │
│  results.md (bảng điểm + phân tích worst performers)       │
└─────────────────────────────────────────────────────────────┘
```

---

## Yêu cầu Evaluation (Checklist)

- [x] **Golden Dataset** — 16 cặp Q&A (`group_project/evaluation/golden_dataset.json`)
- [x] **Script evaluation** — (`group_project/evaluation/eval_pipeline.py`)
- [x] **4 Metrics**: Faithfulness, Answer Relevance, Context Recall, Context Precision
- [x] **So sánh A/B** — Config A (Hybrid+Rerank) vs Config B (Dense-only)
- [x] **Báo cáo** — `group_project/evaluation/results.md`

---

## Cấu Trúc File

```
group_project/
├── README.md                    ← File này
└── evaluation/
    ├── golden_dataset.json      ← 16 cặp Q&A pháp luật ma tuý + nghệ sĩ
    ├── eval_pipeline.py         ← Script evaluation DeepEval đầy đủ
    └── results.md               ← Báo cáo tự động sinh sau khi chạy eval
```

---

## Framework Evaluation

| Framework | Lý do chọn |
|-----------|------------|
| **DeepEval** ✅ | Đã có trong `requirements.txt`, nhiều metrics built-in, hỗ trợ OpenRouter API (OpenAI-compatible), dễ integrate với pytest |
| ~~RAGAS~~ | Alternative — không chọn |
| ~~TruLens~~ | Alternative — không chọn |

---

## Hướng Dẫn Chạy

### 1. Cài đặt dependencies

```bash
pip install deepeval openai python-dotenv sentence-transformers rank-bm25
```

### 2. Kiểm tra file `.env`

```bash
# .env phải có:
OPENROUTER_API_KEY=sk-or-v1-...   # dùng cho cả RAG generation lẫn DeepEval judge
```

### 3. Đảm bảo vector store đã được build

```bash
# Nếu chưa có data/vector_store.pkl
python -m src.task4_chunking_indexing
```

### 4. Chạy Evaluation Pipeline

```bash
# Từ project root
python -m group_project.evaluation.eval_pipeline
```

Pipeline sẽ:
1. Load 16 câu hỏi từ `golden_dataset.json`
2. Chạy RAG với **Config A** (Hybrid + Rerank) → gọi LLM → evaluate
3. Chạy RAG với **Config B** (Dense-only) → gọi LLM → evaluate
4. So sánh A/B và tạo báo cáo `results.md`

### 5. Xem kết quả

```
group_project/evaluation/results.md
```

---

## Configs A/B Test

| | Config A | Config B |
|---|----------|----------|
| **Tên** | Hybrid Search + Reranking | Dense-Only |
| **Semantic Search** | ✅ | ✅ |
| **BM25 Lexical** | ✅ | ❌ |
| **RRF Fusion** | ✅ | ❌ |
| **Cross-Encoder Rerank** | ✅ | ❌ |
| **PageIndex Fallback** | ✅ | ✅ |

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Nguyễn Vũ Trọng | 2A202600960 | Task 1–10 (bài cá nhân) + Thiết lập Eval Pipeline + Golden Dataset | ✅ Hoàn thành |
| Nguyễn Phương Nam | 2A202600962 | Phân tích kết quả Evaluation + So sánh A/B + Báo cáo results.md | ✅ Hoàn thành |
| Hồ Tất Bảo Hoàng | 2A202600699 | Hỗ trợ thu thập dữ liệu + Documentation + Review & kiểm thử pipeline | ✅ Hoàn thành |

---

## Tài Liệu Tham Khảo

- [DeepEval Documentation](https://docs.confident-ai.com/)
- [OpenRouter API](https://openrouter.ai/docs)
- [RAGAS Paper](https://arxiv.org/abs/2309.15217) — Metrics reference
- Liu et al. (2023), *Lost in the Middle: How Language Models Use Long Contexts*
