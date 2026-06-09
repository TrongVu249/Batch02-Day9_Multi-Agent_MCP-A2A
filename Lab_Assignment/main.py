"""
main.py — Demo Script cho Supervisor-Workers Multi-Agent RAG System.

Chạy lệnh:
    python -m Lab_Assignment.src.main
    # hoặc từ thư mục Lab_Assignment:
    python -m src.main

Mô tả:
    Demo hệ thống multi-agent RAG với Supervisor-Workers pattern,
    applied trên RAG pipeline Day08 (pháp luật ma tuý VN + tin tức nghệ sĩ).

Kiến trúc:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                  Supervisor-Workers Multi-Agent RAG                 │
    │                                                                     │
    │   User Query                                                        │
    │       │                                                             │
    │       ▼                                                             │
    │   ┌──────────────────────────────────────┐                         │
    │   │  SUPERVISOR (Rule-based Routing)      │                         │
    │   │  - Phân loại query (legal/news/gen)  │                         │
    │   │  - Chọn mode (hybrid vs dense)       │                         │
    │   │  - Dispatch Workers song song        │                         │
    │   └────────────────┬─────────────────────┘                         │
    │                    │                                                │
    │         ┌──────────┴──────────┐                                     │
    │         ▼                     ▼                                     │
    │  ┌─────────────────┐  ┌──────────────────────┐                     │
    │  │ WORKER 1        │  │ WORKER 2             │                     │
    │  │ Retrieval       │  │ Legal Analysis        │                     │
    │  │                 │  │                      │                     │
    │  │ ① Semantic      │  │ • Phân tích điều     │                     │
    │  │    Search       │  │   luật liên quan     │                     │
    │  │ ② BM25 Lexical  │  │ • Xác định hình      │                     │
    │  │    Search       │  │   phạt, quy trình    │                     │
    │  │ ③ RRF Merge     │  │ • Cung cấp context   │                     │
    │  │ ④ Cross-Encoder │  │   pháp lý chuyên     │                     │
    │  │    Rerank       │  │   sâu cho Worker 3   │                     │
    │  │ ⑤ PageIndex     │  │                      │                     │
    │  │    Fallback     │  │   [LLM: gpt-4o-mini] │                     │
    │  └────────┬────────┘  └──────────┬───────────┘                     │
    │           │                      │                                  │
    │           └──────────┬───────────┘                                  │
    │                      ▼                                              │
    │            ┌─────────────────┐                                      │
    │            │   AGGREGATE     │                                      │
    │            │ Gộp kết quả     │                                      │
    │            └────────┬────────┘                                      │
    │                     ▼                                               │
    │  ┌──────────────────────────────────────┐                           │
    │  │  WORKER 3: Citation & Synthesis      │                           │
    │  │  • Reorder chunks (tránh lost-in-    │                           │
    │  │    the-middle effect)                │                           │
    │  │  • Format context với source labels  │                           │
    │  │  • Tích hợp phân tích pháp lý       │                           │
    │  │  • LLM generation với citation       │                           │
    │  └──────────────────────────────────────┘                           │
    │                     │                                               │
    │                     ▼                                               │
    │           Final Answer có Citation                                  │
    └─────────────────────────────────────────────────────────────────────┘
"""

import asyncio
import logging
import os
import sys
import time

# Smart path setup — works whether run from:
#   1. Day09 root:       python -m Lab_Assignment.main
#   2. Lab_Assignment:   python main.py
#   3. Lab_Assignment:   python -m src.main (if __init__ exists)

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)  # Batch02-Day9... root

# Add project root so 'Lab_Assignment.src' and 'common' are importable
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# Add Lab_Assignment dir so relative 'src' imports work too
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from dotenv import load_dotenv

# Try loading .env from project root first, then current dir
load_dotenv(os.path.join(_project_root, ".env"))
load_dotenv()

# Import: try absolute path first (from project root), then relative (from Lab_Assignment)
try:
    from Lab_Assignment.src.supervisor_rag import ask, create_rag_graph
except ModuleNotFoundError:
    from src.supervisor_rag import ask, create_rag_graph

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_header(title: str, char: str = "=", width: int = 72) -> None:
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_section(title: str, width: int = 72) -> None:
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def print_result(result: dict, elapsed: float) -> None:
    query_type_icon = {
        "legal": "⚖️  Legal",
        "news": "📰 News",
        "general": "🌐 General",
    }.get(result["query_type"], result["query_type"])

    print_section("FINAL ANSWER")
    print(result["answer"])

    print_section("METADATA")
    print(f"  Query type    : {query_type_icon}")
    print(f"  Sources used  : {len(result['sources'])} chunks")
    print(f"  Elapsed time  : {elapsed:.2f}s")

    if result["sources"]:
        print_section("SOURCES")
        for i, src in enumerate(result["sources"][:3], 1):
            meta = src.get("metadata", {})
            print(
                f"  [{i}] {meta.get('source', 'unknown')} "
                f"| type={meta.get('type', '?')} "
                f"| score={src.get('score', 0):.3f}"
            )
        if len(result["sources"]) > 3:
            print(f"  ... và {len(result['sources']) - 3} nguồn khác")


# ---------------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    {
        "label": "Legal Query — Hình phạt tàng trữ ma tuý",
        "question": "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
    },
    {
        "label": "News Query — Nghệ sĩ liên quan ma tuý",
        "question": "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
    },
    {
        "label": "Legal Query — Quy trình cai nghiện",
        "question": "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    },
]


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

async def run_demo() -> None:
    print_header("Supervisor-Workers Multi-Agent RAG System", "═")
    print("""
  Cải tiến Agent Day08 — RAG Pipeline v2
  Áp dụng pattern: Supervisor → [Worker1 + Worker2] → Aggregate → Worker3

  Điểm cải tiến so với monolithic pipeline (task9 + task10):
    ✅ Workers 1 & 2 chạy SONG SONG (giảm latency)
    ✅ Worker 2 bổ sung phân tích pháp lý CHUYÊN SÂU
    ✅ Supervisor ROUTE thông minh: legal/news/general
    ✅ Cấu trúc graph TƯỜNG MINH, dễ debug và mở rộng
    ✅ Supervisor chọn hybrid vs dense-only dựa trên query type
""")

    # Optionally visualise the graph
    try:
        graph = create_rag_graph()
        mermaid = graph.get_graph().draw_mermaid()
        print("  [Graph Topology — Mermaid]")
        for line in mermaid.split("\n")[:15]:
            print(f"    {line}")
        print("    ...")
    except Exception:
        pass

    for i, item in enumerate(TEST_QUERIES, 1):
        print_header(f"Query {i}/{len(TEST_QUERIES)}: {item['label']}", "─")
        print(f"\n  Q: {item['question']}\n")
        print("  [Agent Execution Log]")
        print("  " + "·" * 68)

        start = time.perf_counter()
        try:
            result = await ask(item["question"])
            elapsed = time.perf_counter() - start
            print("  " + "·" * 68)
            print_result(result, elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"\n  ❌ Error: {e}")
            print(f"  (Elapsed: {elapsed:.2f}s)")

        if i < len(TEST_QUERIES):
            print("\n  [Tiếp tục query tiếp theo...]")

    print_header("DEMO HOÀN TẤT", "═")
    print("""
  Supervisor-Workers Pattern Summary:
  ┌──────────────────────────────────────────────────────────┐
  │ Supervisor    │ Rule-based routing, zero LLM overhead    │
  │ Worker 1      │ Retrieval: Semantic+BM25+Rerank+Fallback │
  │ Worker 2      │ Legal Analysis: LLM specialist           │
  │ Worker 3      │ Citation & Synthesis: LLM generation     │
  └──────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    # Allow passing a single custom question via CLI
    if len(sys.argv) > 1:
        custom_question = " ".join(sys.argv[1:])

        async def run_single():
            print_header("Supervisor-Workers Multi-Agent RAG System", "═")
            print(f"\n  Q: {custom_question}\n")
            print("  [Agent Execution Log]")
            print("  " + "·" * 68)

            start = time.perf_counter()
            result = await ask(custom_question)
            elapsed = time.perf_counter() - start

            print("  " + "·" * 68)
            print_result(result, elapsed)

        asyncio.run(run_single())
    else:
        asyncio.run(run_demo())
