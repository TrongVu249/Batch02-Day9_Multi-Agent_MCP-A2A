# Supervisor-Workers Multi-Agent RAG System

**Assignment: Improve Agent Day08 sử dụng pattern Supervisor-Workers (ít nhất 2-3 Workers)**

---

## Tổng Quan

Hệ thống này cải tiến **RAG Pipeline Day08** (pháp luật ma tuý VN + tin tức nghệ sĩ) bằng cách áp dụng **Supervisor-Workers pattern** với LangGraph StateGraph.

### So sánh: Monolithic vs Supervisor-Workers

| Tiêu chí | Monolithic (task9 + task10) | **Supervisor-Workers (Mới)** |
|----------|----------------------------|------------------------------|
| Cấu trúc | Sequential pipeline | Graph với parallel nodes |
| Execution | Tuần tự | **Workers 1 & 2 song song** |
| Specialisation | Một flow duy nhất | **Mỗi worker chuyên biệt** |
| Routing | Cố định | **Supervisor route thông minh** |
| Legal Context | Không có | **Worker 2: Legal Analysis** |
| Debug | Khó trace | **Log rõ từng node** |

---

## Kiến Trúc

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  SUPERVISOR — Rule-based Routing (0ms overhead)              │
│  • Phân loại query: legal / news / general                   │
│  • Chọn mode: hybrid (Semantic+BM25) vs dense-only           │
│  • Dispatch Workers 1 & 2 song song qua LangGraph Send API   │
└──────────────────────────────────────────────────────────────┘
    │
    ├─────────────────────────────────────┐
    ▼                                     ▼
┌────────────────────────┐   ┌───────────────────────────────┐
│  WORKER 1: Retrieval   │   │  WORKER 2: Legal Analysis     │
│                        │   │                               │
│  ① Semantic Search     │   │  • System prompt chuyên sâu   │
│  ② BM25 Lexical Search │   │    về luật ma tuý VN          │
│  ③ RRF Merge           │   │  • Xác định điều luật liên    │
│  ④ Cross-Encoder Rerank│   │    quan (Điều 247-259 BLHS)   │
│  ⑤ PageIndex Fallback  │   │  • Hình phạt, quy trình       │
│                        │   │  • Context pháp lý cho W3     │
└──────────┬─────────────┘   └───────────────┬───────────────┘
           │                                 │
           └─────────────────────────────────┘
                             │
                             ▼
               ┌─────────────────────────┐
               │  AGGREGATE              │
               │  Gộp kết quả từ W1 & W2 │
               └────────────┬────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────────┐
  │  WORKER 3: Citation & Synthesis                         │
  │                                                         │
  │  • Reorder chunks (tránh "lost in the middle" effect)   │
  │  • Format context với source labels rõ ràng             │
  │  • Tích hợp Legal Analysis từ Worker 2                  │
  │  • LLM generation với yêu cầu citation bắt buộc        │
  │    [Nguồn, Năm] cho mọi thông tin                       │
  └─────────────────────────────────────────────────────────┘
                            │
                            ▼
                   Final Answer có Citation
```

---

## Cấu Trúc File

```
Lab_Assignment/
├── __init__.py                    ← Package init
├── main.py                        ← Demo script (chạy được)
├── README_ASSIGNMENT.md           ← File này
├── src/
│   ├── __init__.py
│   ├── supervisor_rag.py          ← ⭐ SUPERVISOR-WORKERS SYSTEM (MỚI)
│   ├── task1_collect_legal_docs.py
│   ├── task2_crawl_news.py
│   ├── task3_convert_markdown.py
│   ├── task4_chunking_indexing.py  ← Vector store builder
│   ├── task5_semantic_search.py    ← Worker 1 tool: dense search
│   ├── task6_lexical_search.py     ← Worker 1 tool: BM25
│   ├── task7_reranking.py          ← Worker 1 tool: cross-encoder
│   ├── task8_pageindex_vectorless.py ← Worker 1 tool: fallback
│   ├── task9_retrieval_pipeline.py  ← Original pipeline (baseline)
│   └── task10_generation.py         ← Original generation (baseline)
└── data/
    ├── landing/                    ← Raw documents
    ├── standardized/               ← Converted markdown
    └── vector_store.pkl            ← Embedded chunks
```

---

## Cách Chạy

### 1. Setup environment

```bash
# Từ thư mục Day09 project root
cd "d:\Project\Vin_AI\Lab 9\Batch02-Day9_Multi-Agent_MCP-A2A"

# Đảm bảo .env có OPENROUTER_API_KEY
# OPENROUTER_API_KEY=sk-or-v1-...
# OPENROUTER_MODEL=openai/gpt-4o-mini
```

### 2. Chạy Demo (3 test queries)

```bash
# Từ Day09 project root:
.venv\Scripts\python.exe -m Lab_Assignment.main

# Hoặc từ thư mục Lab_Assignment:
python main.py
```

### 3. Hỏi câu hỏi tùy chỉnh

```bash
.venv\Scripts\python.exe -m Lab_Assignment.main "Hình phạt tội mua bán ma tuý?"
```

### 4. Dùng như Python API

```python
import asyncio
from Lab_Assignment.src.supervisor_rag import ask

result = asyncio.run(ask("Hình phạt cho tội tàng trữ trái phép chất ma tuý?"))

print(result["answer"])          # Câu trả lời có citation
print(result["query_type"])      # 'legal' | 'news' | 'general'
print(len(result["sources"]))    # Số chunks nguồn
```

---

## Agents Chi Tiết

### Supervisor (Rule-based Routing)

- **Không gọi LLM** → zero overhead
- Phân loại query bằng keyword matching
- `legal` → hybrid mode + Legal Analysis Worker
- `news` → dense-only mode (không cần BM25)
- `general` → hybrid + Legal Analysis

### Worker 1: Retrieval (task5–task9 modules)

Tools sử dụng:
| Tool | Function |
|------|----------|
| `semantic_search()` | Dense vector similarity (cosine) |
| `lexical_search()` | BM25 keyword search |
| `rerank_rrf()` | Reciprocal Rank Fusion merge |
| `rerank()` | Cross-Encoder reranking |
| `pageindex_search()` | Vectorless fallback |

### Worker 2: Legal Analysis (LLM Specialist)

- System prompt chuyên sâu về pháp luật ma tuý VN
- Identify Điều 247-259 BLHS 2015, Luật 73/2021/QH15
- Output: phân tích ngắn gọn (<150 từ) cho Worker 3

### Worker 3: Citation & Synthesis (LLM Generation)

- Reorder chunks: tránh "lost in the middle" effect
  - Pattern: `[1, 3, 5, 4, 2]` thay vì `[1, 2, 3, 4, 5]`
- Format context với source labels rõ ràng
- Tích hợp Legal Analysis từ Worker 2
- Yêu cầu LLM chèn citation `[Nguồn, Năm]`

---

## Cải Tiến So Với Day08 Gốc

| Aspect | Trước (Monolithic) | Sau (Supervisor-Workers) |
|--------|-------------------|--------------------------|
| **Architecture** | Single function call | Multi-agent StateGraph |
| **Parallelism** | Sequential | Worker 1 + 2 run concurrently |
| **Routing** | Hardcoded | Supervisor classifies & routes |
| **Legal Context** | None | Worker 2 provides expert analysis |
| **Observability** | No logs | Step-by-step agent logs |
| **Extensibility** | Hard to add | Add new worker = add node |
| **Mode** | Always hybrid | Supervisor selects best mode |

---

## Dependencies

Tái sử dụng các packages đã có trong Day09 project:
- `langgraph` — StateGraph, Send API
- `langchain-openai` — ChatOpenAI via OpenRouter
- `langchain-core` — Messages

Thêm cho Day08 modules (cần install vào venv):
- `numpy` — Vector operations
- `sentence-transformers` — Embedding + CrossEncoder
- `rank-bm25` — BM25 lexical search
