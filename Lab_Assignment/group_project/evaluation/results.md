# RAG Evaluation Results

## Framework sử dụng

> **DeepEval** — được chọn vì nhiều metrics built-in, dễ integrate với pytest,
> và hỗ trợ OpenRouter API (OpenAI-compatible format).

> LLM judge: `gpt-4o-mini` qua OpenRouter API.

---

## Overall Scores

| Metric | Config A — Hybrid + Rerank | Config B — Dense-Only | Δ (A−B) |
|--------|---------------------------|----------------------|---------|
| Faithfulness | 0.986 | 0.970 | +0.016 |
| Answer Relevance | 0.334 | 0.263 | +0.070 |
| Context Recall | 0.797 | 0.688 | +0.109 |
| Context Precision | 0.311 | 0.319 | -0.008 |
| **Average** | **0.607** | **0.560** | **+0.047** |

---

## A/B Comparison Analysis

**Config A — Hybrid Search + Reranking:**
> - Semantic search (dense) + BM25 lexical search → RRF fusion
> - Cross-encoder reranking để chấm lại relevance
> - PageIndex fallback nếu score < threshold

**Config B — Dense-Only (No Reranking):**
> - Chỉ semantic search (cosine similarity)
> - Không có lexical search, không có reranking
> - Trả về top-k theo embedding score trực tiếp

**Kết luận:** Config A đạt điểm trung bình cao hơn (0.047 điểm, 
tức +8.4%). 
Kết hợp hybrid search + reranking giúp tăng chất lượng retrieval và độ chính xác
của câu trả lời, đặc biệt cải thiện Context Recall và Faithfulness.

---

## Worst Performers (Bottom 3)

*Phân tích từ Config A — Hybrid + Reranking*

| # | Question | Faith. | Relevance | Recall | Precision | Failure Stage | Root Cause |
|---|----------|--------|-----------|--------|-----------|---------------|------------|
| 1 | Luật Phòng chống ma tuý 2021 quy định những hình t... | 1.00 | 0.00 | 0.00 | 0.00 | Retrieval | Retriever không lấy đủ evidence cho câu hỏi này |
| 2 | Tội sử dụng trái phép chất ma tuý theo Bộ luật Hìn... | 1.00 | 0.00 | 0.00 | 0.00 | Retrieval | Retriever không lấy đủ evidence cho câu hỏi này |
| 3 | Mức xử phạt hành chính đối với hành vi sử dụng trá... | 1.00 | 0.00 | 0.00 | 0.00 | Retrieval | Retriever không lấy đủ evidence cho câu hỏi này |

---

## Recommendations

### Cải tiến 1: Dùng embedding model tiếng Việt
**Action:** Thay `all-MiniLM-L6-v2` bằng `BAAI/bge-m3` hoặc `keepitreal/vietnamese-sbert`
để tăng chất lượng embedding cho văn bản pháp luật tiếng Việt.
**Expected impact:** Tăng Context Recall và Faithfulness thêm 10–15%.

### Cải tiến 2: Tăng kích thước chunk và overlap
**Action:** Tăng `chunk_size` từ 500 lên 800–1000 ký tự, `overlap` từ 50 lên 150.
Mỗi chunk sẽ chứa đủ context pháp lý (điều khoản + phần giải thích liền kề).
**Expected impact:** Giảm trường hợp câu trả lời thiếu thông tin (Context Recall +8%).

### Cải tiến 3: Dùng Jina Reranker multilingual
**Action:** Cung cấp JINA_API_KEY hợp lệ để kích hoạt `jina-reranker-v2-base-multilingual`
thay vì local cross-encoder `ms-marco-MiniLM-L-6-v2` (chỉ tốt cho tiếng Anh).
**Expected impact:** Reranking chính xác hơn cho tiếng Việt, tăng Context Precision +12%.

---

*Báo cáo được tạo tự động bởi `eval_pipeline.py`*