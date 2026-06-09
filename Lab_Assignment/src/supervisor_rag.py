"""
Supervisor-Workers Multi-Agent RAG System — Day08 Improvement.

Kiến trúc:
    Supervisor (RAG Orchestrator)
        ├── Worker 1: Retrieval Worker   (semantic + lexical + rerank + pageindex fallback)
        ├── Worker 2: Legal Analysis Worker  (chuyên gia pháp luật ma tuý VN)
        └── Worker 3: Citation & Synthesis Worker  (tổng hợp + generate có citation)

Graph topology:
    supervisor_route
        → [retrieval_worker  (parallel)
           legal_analysis_worker (parallel)]
        → aggregate_results
        → synthesis_worker
        → END

Điểm cải tiến so với monolithic pipeline (task9 + task10):
    + Parallel execution: Worker 1 & 2 chạy đồng thời
    + Specialisation: mỗi worker có system prompt và tools riêng
    + Legal context: Worker 2 bổ sung phân tích pháp lý chuyên sâu
      trước khi sinh câu trả lời
    + Structured flow: explicit graph topology, dễ debug và mở rộng
    + Intelligent routing: Supervisor phân loại query và chọn mode phù hợp
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Send
from langgraph.graph import END, StateGraph

# ---------------------------------------------------------------------------
# Load environment variables
# — Try project root first (Day09), then current dir
# ---------------------------------------------------------------------------

_this_dir = os.path.dirname(os.path.abspath(__file__))
_lab_dir = os.path.dirname(_this_dir)          # Lab_Assignment/
_project_root = os.path.dirname(_lab_dir)      # Batch02-Day9.../

load_dotenv(os.path.join(_project_root, ".env"))
load_dotenv()  # Fallback to current dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------

_llm_instance = None


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter (shared with Day09)."""
    global _llm_instance
    if _llm_instance is None:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in .env")
        _llm_instance = ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.3,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )
    return _llm_instance


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------

def _last_wins(a, b):
    """Reducer: keep the most recently written value (for parallel branches)."""
    return b if b is not None else a


class RAGState(TypedDict):
    """Shared state cho toàn bộ multi-agent RAG graph."""

    # Input
    question: str                                           # Câu hỏi gốc của user

    # Supervisor decision
    query_type: str                                         # 'legal' | 'news' | 'general'
    use_lexical: bool                                       # hybrid vs dense-only mode
    needs_legal_analysis: bool                              # Có cần Worker 2 không

    # Worker 1: Retrieval results
    retrieval_results: Annotated[list, _last_wins]          # Chunks từ vector store

    # Worker 2: Legal analysis
    legal_analysis: Annotated[str, _last_wins]              # Phân tích pháp lý chuyên sâu

    # Worker 3: Final output
    final_answer: str                                       # Câu trả lời có citation
    sources: list                                           # Nguồn đã dùng


# ---------------------------------------------------------------------------
# SUPERVISOR NODE
# ---------------------------------------------------------------------------

async def supervisor_route(state: RAGState) -> dict:
    """
    Supervisor — Phân loại query và quyết định routing.

    Sử dụng rule-based routing (keyword matching) để:
    1. Phân loại query: legal / news / general
    2. Quyết định mode: hybrid (semantic + BM25) vs dense-only
    3. Quyết định có cần Worker 2 (Legal Analysis) không

    Không gọi LLM → 0ms overhead → nhanh hơn.
    """
    question = state["question"].lower()

    # --- Classify query type ---
    legal_keywords = [
        "luật", "điều", "khoản", "hình phạt", "tội", "bộ luật", "nghị định",
        "thông tư", "pháp luật", "cai nghiện", "phạt tù", "phạt tiền",
        "tàng trữ", "mua bán", "vận chuyển", "sản xuất", "tội phạm", "xử lý",
        "law", "penalty", "statute", "regulation", "criminal",
    ]
    news_keywords = [
        "nghệ sĩ", "ca sĩ", "diễn viên", "người nổi tiếng", "celeb",
        "bị bắt", "bị khởi tố", "scandal", "vụ án", "cảnh sát", "triệt phá",
        "artist", "celebrity", "arrested",
    ]

    legal_score = sum(1 for kw in legal_keywords if kw in question)
    news_score = sum(1 for kw in news_keywords if kw in question)

    if legal_score > news_score:
        query_type = "legal"
        use_lexical = True          # BM25 tốt cho từ khóa pháp lý chính xác
        needs_legal_analysis = True  # Luôn cần phân tích pháp lý cho câu hỏi luật
    elif news_score > 0:
        query_type = "news"
        use_lexical = False         # Dense search tốt hơn cho ngữ nghĩa bài báo
        needs_legal_analysis = False
    else:
        query_type = "general"
        use_lexical = True          # Hybrid cho câu hỏi chung
        needs_legal_analysis = True  # Mặc định có phân tích pháp lý

    logger.info(
        "[Supervisor] Query type: %s | use_lexical: %s | needs_legal_analysis: %s",
        query_type, use_lexical, needs_legal_analysis,
    )
    print(f"\n[Supervisor] 🎯 Phân loại query: '{query_type}'")
    print(f"[Supervisor] ⚙️  Mode: {'hybrid (semantic + BM25)' if use_lexical else 'dense-only'}")
    print(f"[Supervisor] 🔀 Workers: Retrieval + {'Legal Analysis' if needs_legal_analysis else '(skip Legal Analysis)'}")

    return {
        "query_type": query_type,
        "use_lexical": use_lexical,
        "needs_legal_analysis": needs_legal_analysis,
        "retrieval_results": [],
        "legal_analysis": "",
        "final_answer": "",
        "sources": [],
    }


# ---------------------------------------------------------------------------
# ROUTING FUNCTION (dispatches Workers 1 & 2 in parallel)
# ---------------------------------------------------------------------------

def route_to_workers(state: RAGState) -> list[Send]:
    """
    Routing function: dispatch Worker 1 + (optionally) Worker 2 in parallel.

    Uses LangGraph's Send API để chạy workers đồng thời.
    """
    sends = []

    # Worker 1: Retrieval luôn chạy
    sends.append(Send("retrieval_worker", state))

    # Worker 2: Legal Analysis — chỉ chạy khi cần
    if state.get("needs_legal_analysis", True):
        sends.append(Send("legal_analysis_worker", state))

    return sends


# ---------------------------------------------------------------------------
# WORKER 1: RETRIEVAL WORKER
# ---------------------------------------------------------------------------

async def retrieval_worker(state: RAGState) -> dict:
    """
    Worker 1 — Retrieval Specialist.

    Thực hiện toàn bộ retrieval pipeline:
        1. Semantic Search (dense vector similarity)
        2. Lexical Search - BM25 (nếu hybrid mode)
        3. RRF Merge (nếu hybrid)
        4. Cross-Encoder Rerank
        5. PageIndex fallback (nếu score thấp)

    Tái sử dụng trực tiếp các modules Day08 (task5–task9).
    """
    print("\n[Worker 1: Retrieval] 🔍 Bắt đầu tìm kiếm...")

    question = state["question"]
    use_lexical = state.get("use_lexical", True)
    top_k = 5
    score_threshold = 0.3

    try:
        # Try relative imports first (when run as part of Lab_Assignment package)
        try:
            from .task5_semantic_search import semantic_search
            from .task6_lexical_search import lexical_search
            from .task7_reranking import rerank, rerank_rrf
            from .task8_pageindex_vectorless import pageindex_search
        except ImportError:
            # Fallback to absolute imports (when src/ is on sys.path)
            from task5_semantic_search import semantic_search  # type: ignore
            from task6_lexical_search import lexical_search    # type: ignore
            from task7_reranking import rerank, rerank_rrf     # type: ignore
            from task8_pageindex_vectorless import pageindex_search  # type: ignore

        # Step 1: Semantic Search (luôn chạy)
        print(f"[Worker 1: Retrieval]   → Semantic Search (top_k={top_k * 2})...")
        dense_results = semantic_search(question, top_k=top_k * 2)
        print(f"[Worker 1: Retrieval]   ✓ Semantic: {len(dense_results)} kết quả")

        if use_lexical:
            # Step 2: Lexical Search (BM25) — Hybrid mode
            print("[Worker 1: Retrieval]   → Lexical Search (BM25)...")
            sparse_results = lexical_search(question, top_k=top_k * 2)
            print(f"[Worker 1: Retrieval]   ✓ BM25: {len(sparse_results)} kết quả")

            # Step 3: RRF Merge
            merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
            for item in merged:
                item["source"] = "hybrid"
            print(f"[Worker 1: Retrieval]   ✓ RRF Merge: {len(merged)} kết quả")
        else:
            # Dense-only mode
            merged = dense_results[:top_k * 2]
            for item in merged:
                item["source"] = "dense"

        # Step 4: Cross-Encoder Rerank
        if merged:
            print("[Worker 1: Retrieval]   → Cross-Encoder Rerank...")
            final_results = rerank(question, merged, top_k=top_k, method="cross_encoder")
            print(f"[Worker 1: Retrieval]   ✓ Reranked: {len(final_results)} kết quả")
        else:
            final_results = []

        # Step 5: PageIndex Fallback
        if not final_results or (final_results and final_results[0].get("score", 0) < score_threshold):
            print("[Worker 1: Retrieval]   → Kết quả chưa đủ tốt, thử PageIndex fallback...")
            try:
                fallback = pageindex_search(question, top_k=top_k)
                if fallback:
                    final_results = fallback
                    print(f"[Worker 1: Retrieval]   ✓ PageIndex: {len(fallback)} kết quả")
            except Exception as e:
                print(f"[Worker 1: Retrieval]   ⚠ PageIndex unavailable: {e}")

        print(f"[Worker 1: Retrieval] ✅ Hoàn tất: {len(final_results)} chunks")
        return {"retrieval_results": final_results[:top_k]}

    except Exception as e:
        logger.exception("[Worker 1: Retrieval] Error: %s", e)
        print(f"[Worker 1: Retrieval] ❌ Lỗi: {e}")
        return {"retrieval_results": []}


# ---------------------------------------------------------------------------
# WORKER 2: LEGAL ANALYSIS WORKER
# ---------------------------------------------------------------------------

LEGAL_EXPERT_PROMPT = """Bạn là chuyên gia pháp lý chuyên về **pháp luật Việt Nam về ma tuý và các chất cấm**.

Kiến thức chuyên sâu của bạn bao gồm:
- Luật Phòng, chống ma tuý 2021 (Luật số 73/2021/QH15)
- Bộ luật Hình sự 2015 (sửa đổi 2017) — Chương XX: Các tội phạm về ma tuý (Điều 247-259)
- Nghị định 105/2021/NĐ-CP hướng dẫn thi hành Luật Phòng chống ma tuý
- Thông tư liên tịch về danh mục chất ma tuý và tiền chất
- Quy trình cai nghiện bắt buộc và tự nguyện

**Nhiệm vụ**: Phân tích câu hỏi và xác định:
1. Các điều luật liên quan (số điều, khoản)
2. Mức hình phạt áp dụng (nếu là câu hỏi về hình phạt)
3. Quy trình/thủ tục liên quan (nếu là câu hỏi về quy trình)
4. Các khái niệm pháp lý quan trọng cần làm rõ

Giữ phân tích ngắn gọn (dưới 150 từ). Dùng tiếng Việt.
KHÔNG tự bịa thông tin — chỉ phân tích những gì bạn biết chắc."""


async def legal_analysis_worker(state: RAGState) -> dict:
    """
    Worker 2 — Legal Analysis Specialist.

    Phân tích câu hỏi từ góc độ pháp lý chuyên sâu về luật ma tuý VN.
    Cung cấp context pháp lý để Worker 3 tổng hợp câu trả lời tốt hơn.

    Chạy song song với Worker 1 (không phụ thuộc nhau).
    """
    print("\n[Worker 2: Legal Analysis] ⚖️  Phân tích pháp lý...")

    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=LEGAL_EXPERT_PROMPT),
            HumanMessage(content=state["question"]),
        ]
        result = await llm.ainvoke(messages)
        analysis = result.content
        print(f"[Worker 2: Legal Analysis] ✅ Hoàn tất ({len(analysis)} ký tự)")
        return {"legal_analysis": analysis}

    except Exception as e:
        logger.exception("[Worker 2: Legal Analysis] Error: %s", e)
        print(f"[Worker 2: Legal Analysis] ❌ Lỗi: {e}")
        return {"legal_analysis": ""}


# ---------------------------------------------------------------------------
# AGGREGATE NODE
# ---------------------------------------------------------------------------

async def aggregate_results(state: RAGState) -> dict:
    """
    Aggregate Node — Gộp kết quả từ Workers 1 & 2.

    Không gọi LLM ở bước này — chỉ kiểm tra và log kết quả.
    Việc tổng hợp thực sự xảy ra ở Worker 3.
    """
    retrieval_results = state.get("retrieval_results", [])
    legal_analysis = state.get("legal_analysis", "")

    print(f"\n[Aggregate] 📊 Gộp kết quả:")
    print(f"[Aggregate]   Retrieval chunks: {len(retrieval_results)}")
    print(f"[Aggregate]   Legal analysis: {'✓' if legal_analysis else '✗'}")

    # Pass through — no changes needed, state already has both results
    return {}


# ---------------------------------------------------------------------------
# WORKER 3: CITATION & SYNTHESIS WORKER
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = """Trả lời câu hỏi một cách toàn diện bằng **tiếng Việt**.

Với mọi thông tin sự kiện hoặc khẳng định, hãy chèn ngay citation trong ngoặc vuông
liên kết đến nguồn cụ thể. Ví dụ: [Luật Phòng chống ma tuý 2021, Điều 3]
hoặc [VnExpress, 2024] hoặc [Document 1].

Nếu thông tin không có trong context được cung cấp, hãy nói:
'Tôi không thể xác minh thông tin này từ nguồn hiện có.'

Quy tắc:
- Chỉ sử dụng thông tin từ context được cung cấp
- Mọi khẳng định thực tế PHẢI có citation
- Nếu context không đủ, nói rõ điều đó
- Trình bày câu trả lời rõ ràng theo đoạn văn
- Tích hợp Phân tích Pháp lý vào câu trả lời khi phù hợp"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh 'lost in the middle' effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt.
    Strategy: chunks quan trọng nhất → đầu và cuối, kém hơn → giữa.

    Input (by score):  [1, 2, 3, 4, 5]
    Output:            [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return chunks
    reordered = []
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])
    evens = [chunks[i] for i in range(1, len(chunks), 2)]
    reordered.extend(evens[::-1])
    return reordered


def format_context(chunks: list[dict], legal_analysis: str) -> str:
    """
    Format chunks + legal analysis thành context string cho prompt.

    Mỗi chunk có label source để LLM có thể cite chính xác.
    """
    context_parts = []

    # Legal analysis từ Worker 2 (nếu có)
    if legal_analysis:
        context_parts.append(
            f"[Phân Tích Pháp Lý — Chuyên Gia]\n{legal_analysis}"
        )

    # Retrieval chunks từ Worker 1
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        score = chunk.get("score", 0.0)
        retrieval_source = chunk.get("source", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type} "
            f"| Score: {score:.3f} | Retrieved via: {retrieval_source}]\n"
            f"{chunk['content']}"
        )

    return "\n\n---\n\n".join(context_parts)


async def synthesis_worker(state: RAGState) -> dict:
    """
    Worker 3 — Citation & Synthesis Specialist.

    Tổng hợp retrieval results + legal analysis thành câu trả lời cuối cùng:
        1. Reorder chunks để tránh lost in the middle
        2. Format context với labels nguồn rõ ràng
        3. Tích hợp phân tích pháp lý từ Worker 2
        4. Gọi LLM với SYNTHESIS_SYSTEM_PROMPT (yêu cầu citation)
        5. Trả về câu trả lời + danh sách nguồn
    """
    print("\n[Worker 3: Synthesis] ✍️  Tổng hợp câu trả lời...")

    retrieval_results = state.get("retrieval_results", [])
    legal_analysis = state.get("legal_analysis", "")
    question = state["question"]

    # Step 1: Reorder chunks (tránh lost in the middle)
    reordered = reorder_for_llm(retrieval_results)
    print(f"[Worker 3: Synthesis]   Reordered {len(reordered)} chunks")

    # Step 2: Format context với sources labels
    context = format_context(reordered, legal_analysis)

    # Step 3: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nCâu hỏi: {question}"

    try:
        # Step 4: Call LLM
        llm = get_llm()
        messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        result = await llm.ainvoke(messages)
        answer = result.content

        print(f"[Worker 3: Synthesis] ✅ Hoàn tất ({len(answer)} ký tự)")

        return {
            "final_answer": answer,
            "sources": retrieval_results,
        }

    except Exception as e:
        logger.exception("[Worker 3: Synthesis] Error: %s", e)
        print(f"[Worker 3: Synthesis] ❌ Lỗi: {e}")
        return {
            "final_answer": f"Lỗi khi tổng hợp câu trả lời: {e}",
            "sources": retrieval_results,
        }


# ---------------------------------------------------------------------------
# GRAPH CONSTRUCTION
# ---------------------------------------------------------------------------

def create_rag_graph():
    """
    Build và compile multi-agent RAG StateGraph.

    Topology:
        supervisor_route
            → [retrieval_worker  ─── (parallel via Send API)
               legal_analysis_worker]
            → aggregate_results
            → synthesis_worker
            → END
    """
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("supervisor_route", supervisor_route)
    graph.add_node("retrieval_worker", retrieval_worker)
    graph.add_node("legal_analysis_worker", legal_analysis_worker)
    graph.add_node("aggregate_results", aggregate_results)
    graph.add_node("synthesis_worker", synthesis_worker)

    # Set entry point
    graph.set_entry_point("supervisor_route")

    # Supervisor → parallel dispatch to Workers 1 & 2
    graph.add_conditional_edges(
        "supervisor_route",
        route_to_workers,
        ["retrieval_worker", "legal_analysis_worker"],
    )

    # Both workers → aggregate
    graph.add_edge("retrieval_worker", "aggregate_results")
    graph.add_edge("legal_analysis_worker", "aggregate_results")

    # Aggregate → Worker 3
    graph.add_edge("aggregate_results", "synthesis_worker")

    # Worker 3 → END
    graph.add_edge("synthesis_worker", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

async def ask(question: str) -> dict:
    """
    Public API — Hỏi câu hỏi và nhận câu trả lời có citation.

    Chạy toàn bộ Supervisor-Workers pipeline:
        Supervisor → [Worker1 + Worker2] → Aggregate → Worker3

    Args:
        question: Câu hỏi của user (tiếng Việt)

    Returns:
        {
            'answer': str,          # Câu trả lời có citation
            'sources': list[dict],  # Danh sách chunks đã dùng
            'query_type': str,      # 'legal' | 'news' | 'general'
        }
    """
    graph = create_rag_graph()

    initial_state = {
        "question": question,
        "query_type": "",
        "use_lexical": True,
        "needs_legal_analysis": True,
        "retrieval_results": [],
        "legal_analysis": "",
        "final_answer": "",
        "sources": [],
    }

    result = await graph.ainvoke(initial_state)

    return {
        "answer": result.get("final_answer", ""),
        "sources": result.get("sources", []),
        "query_type": result.get("query_type", "general"),
    }


# ---------------------------------------------------------------------------
# Direct run (for quick test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.WARNING)

    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?"
    )

    print("=" * 70)
    print("Supervisor-Workers Multi-Agent RAG System")
    print("=" * 70)
    print(f"\nQ: {question}\n")
    print("-" * 70)

    result = asyncio.run(ask(question))

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result["answer"])
    print(f"\n[Query type: {result['query_type']} | Sources: {len(result['sources'])} chunks]")
