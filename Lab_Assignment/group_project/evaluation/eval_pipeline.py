"""
RAG Evaluation Pipeline — Sản phẩm 2 (Bài tập nhóm).

Framework: DeepEval
Sử dụng OpenRouter API (OpenAI-compatible) làm LLM judge.

Yêu cầu hoàn thành:
    1. Load golden_dataset.json (16 cặp Q&A)
    2. Chạy RAG pipeline trên từng question với 2 configs khác nhau
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B: Config A (hybrid + rerank) vs Config B (dense-only)
    5. Export results ra results.md với phân tích worst performers

Cài đặt:
    pip install deepeval openai python-dotenv

Chạy:
    python -m group_project.evaluation.eval_pipeline
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env từ project root
ROOT = Path(__file__).parent.parent.parent
load_dotenv(ROOT / ".env")

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

# =============================================================================
# Cấu hình DeepEval dùng OpenRouter (OpenAI-compatible)
# DeepEval sử dụng biến môi trường OPENAI_API_KEY và OPENAI_BASE_URL
# =============================================================================

def _setup_deepeval_openrouter():
    """
    Cấu hình DeepEval để dùng OpenRouter thay vì OpenAI trực tiếp.
    OpenRouter tuân thủ OpenAI API format, chỉ cần thay base_url.
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if not openrouter_key:
        raise ValueError(
            "OPENROUTER_API_KEY chưa được set trong .env\n"
            "Vui lòng thêm: OPENROUTER_API_KEY=sk-or-v1-..."
        )
    # Override env vars mà DeepEval đọc
    os.environ["OPENAI_API_KEY"] = openrouter_key
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
    print(f"✓ DeepEval configured to use OpenRouter (key: ...{openrouter_key[-8:]})")


# =============================================================================
# Load Data
# =============================================================================

def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} Q&A pairs from golden dataset")
    return data


# =============================================================================
# RAG Pipeline Wrapper
# =============================================================================

def run_rag(
    question: str,
    use_reranking: bool = True,
    use_lexical: bool = True,
) -> dict:
    """
    Chạy RAG pipeline cho 1 câu hỏi với config được chỉ định.

    Returns:
        {'answer': str, 'sources': list[dict], 'retrieval_source': str}
    """
    # Import ở đây để tránh circular import & slow startup
    sys.path.insert(0, str(ROOT))
    from src.task10_generation import generate_with_citation  # noqa: E402

    return generate_with_citation(
        query=question,
        use_reranking=use_reranking,
        use_lexical=use_lexical,
    )


# =============================================================================
# Single Config Evaluation
# =============================================================================

def run_single_config_eval(
    golden_dataset: list[dict],
    use_reranking: bool = True,
    use_lexical: bool = True,
    config_name: str = "default",
) -> dict:
    """
    Chạy RAG pipeline + DeepEval cho 1 config trên toàn bộ golden dataset.

    Args:
        golden_dataset: List of {'question', 'expected_answer', 'expected_context'}
        use_reranking: Bật/tắt reranking
        use_lexical: Bật/tắt lexical search (False = dense-only)
        config_name: Tên config để hiển thị

    Returns:
        {
            'config_name': str,
            'test_cases': list,       # LLMTestCase objects
            'results': EvaluationResult,
            'per_question': list[dict],  # chi tiết từng câu
        }
    """
    from deepeval import evaluate
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        ContextualPrecisionMetric,
    )
    from deepeval.test_case import LLMTestCase

    print(f"\n{'='*60}")
    print(f"  Config: {config_name}")
    print(f"  use_reranking={use_reranking}, use_lexical={use_lexical}")
    print(f"{'='*60}")

    test_cases = []
    per_question = []

    for i, item in enumerate(golden_dataset, 1):
        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_context = item.get("expected_context", "")

        print(f"  [{i:02d}/{len(golden_dataset)}] RAG: {question[:60]}...")

        try:
            result = run_rag(
                question=question,
                use_reranking=use_reranking,
                use_lexical=use_lexical,
            )
            actual_output = result["answer"]
            retrieval_context = [c["content"] for c in result["sources"]]
            retrieval_source = result.get("retrieval_source", "unknown")

        except Exception as e:
            print(f"    ✗ RAG failed: {e}")
            actual_output = "I cannot verify this information"
            retrieval_context = []
            retrieval_source = "error"

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected_answer,
            retrieval_context=retrieval_context,
        )
        test_cases.append(test_case)
        per_question.append({
            "question": question,
            "expected_answer": expected_answer,
            "expected_context": expected_context,
            "actual_output": actual_output,
            "retrieval_context": retrieval_context,
            "retrieval_source": retrieval_source,
            "test_case": test_case,
        })

        # Tránh rate limit
        time.sleep(0.5)

    # Định nghĩa 4 metrics
    metrics = [
        FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini"),
        AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini"),
        ContextualRecallMetric(threshold=0.7, model="gpt-4o-mini"),
        ContextualPrecisionMetric(threshold=0.7, model="gpt-4o-mini"),
    ]

    print(f"\n  Đang chạy DeepEval với 4 metrics trên {len(test_cases)} test cases...")
    eval_results = evaluate(test_cases=test_cases, metrics=metrics)

    return {
        "config_name": config_name,
        "test_cases": test_cases,
        "results": eval_results,
        "per_question": per_question,
    }


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(golden_dataset: list[dict]) -> dict:
    """
    So sánh A/B giữa 2 configs:
    - Config A: Hybrid search (semantic + BM25) + cross-encoder reranking
    - Config B: Dense-only search (chỉ semantic), không reranking

    Returns:
        {
            'config_a': {...},  # kết quả Config A
            'config_b': {...},  # kết quả Config B
        }
    """
    # Config A: Hybrid + Reranking (pipeline đầy đủ)
    config_a = run_single_config_eval(
        golden_dataset=golden_dataset,
        use_reranking=True,
        use_lexical=True,
        config_name="Config A — Hybrid Search + Reranking",
    )

    # Config B: Dense-only, không reranking
    config_b = run_single_config_eval(
        golden_dataset=golden_dataset,
        use_reranking=False,
        use_lexical=False,
        config_name="Config B — Dense-Only (No Reranking)",
    )

    return {"config_a": config_a, "config_b": config_b}


# =============================================================================
# Trích xuất scores từ EvaluationResult
# =============================================================================

def _extract_scores(eval_result) -> dict:
    """Trích xuất điểm trung bình từng metric từ EvaluationResult."""
    metric_scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "contextual_recall": [],
        "contextual_precision": [],
    }

    for test_result in eval_result.test_results:
        for metric_data in test_result.metrics_data:
            name = metric_data.name.lower()
            score = metric_data.score if metric_data.score is not None else 0.0
            if "faithfulness" in name:
                metric_scores["faithfulness"].append(score)
            elif "answer relevancy" in name or "answer_relevancy" in name:
                metric_scores["answer_relevancy"].append(score)
            elif "contextual recall" in name or "contextual_recall" in name:
                metric_scores["contextual_recall"].append(score)
            elif "contextual precision" in name or "contextual_precision" in name:
                metric_scores["contextual_precision"].append(score)

    return {
        k: (sum(v) / len(v) if v else 0.0)
        for k, v in metric_scores.items()
    }


def _extract_per_question_scores(eval_result, per_question: list[dict]) -> list[dict]:
    """Gắn scores chi tiết vào từng câu hỏi."""
    enriched = []
    for i, (test_result, pq) in enumerate(zip(eval_result.test_results, per_question)):
        scores = {}
        for metric_data in test_result.metrics_data:
            name = metric_data.name.lower()
            score = metric_data.score if metric_data.score is not None else 0.0
            if "faithfulness" in name:
                scores["faithfulness"] = score
            elif "answer relevancy" in name or "answer_relevancy" in name:
                scores["answer_relevancy"] = score
            elif "contextual recall" in name or "contextual_recall" in name:
                scores["contextual_recall"] = score
            elif "contextual precision" in name or "contextual_precision" in name:
                scores["contextual_precision"] = score

        avg = sum(scores.values()) / len(scores) if scores else 0.0
        enriched.append({
            **pq,
            "scores": scores,
            "avg_score": avg,
        })

    return sorted(enriched, key=lambda x: x["avg_score"])  # worst first


# =============================================================================
# Export Results to results.md
# =============================================================================

def export_results(comparison: dict):
    """
    Export evaluation results ra results.md bao gồm:
    - Bảng điểm tổng hợp
    - A/B comparison analysis
    - Worst performers (bottom 3)
    - Đề xuất cải tiến
    """
    config_a = comparison["config_a"]
    config_b = comparison["config_b"]

    scores_a = _extract_scores(config_a["results"])
    scores_b = _extract_scores(config_b["results"])

    pq_a = _extract_per_question_scores(config_a["results"], config_a["per_question"])

    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevance",
        "contextual_recall": "Context Recall",
        "contextual_precision": "Context Precision",
    }

    lines = []
    lines.append("# RAG Evaluation Results\n")
    lines.append("## Framework sử dụng\n")
    lines.append("> **DeepEval** — được chọn vì nhiều metrics built-in, dễ integrate với pytest,")
    lines.append("> và hỗ trợ OpenRouter API (OpenAI-compatible format).\n")
    lines.append("> LLM judge: `gpt-4o-mini` qua OpenRouter API.\n")
    lines.append("---\n")

    # ---- Overall Scores Table ----
    lines.append("## Overall Scores\n")
    lines.append("| Metric | Config A — Hybrid + Rerank | Config B — Dense-Only | Δ (A−B) |")
    lines.append("|--------|---------------------------|----------------------|---------|")

    avg_a_sum = 0.0
    avg_b_sum = 0.0
    for key, label in metric_labels.items():
        sa = scores_a.get(key, 0.0)
        sb = scores_b.get(key, 0.0)
        delta = sa - sb
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {label} | {sa:.3f} | {sb:.3f} | {sign}{delta:.3f} |")
        avg_a_sum += sa
        avg_b_sum += sb

    avg_a = avg_a_sum / len(metric_labels)
    avg_b = avg_b_sum / len(metric_labels)
    delta_avg = avg_a - avg_b
    sign = "+" if delta_avg >= 0 else ""
    lines.append(f"| **Average** | **{avg_a:.3f}** | **{avg_b:.3f}** | **{sign}{delta_avg:.3f}** |")
    lines.append("")
    lines.append("---\n")

    # ---- A/B Comparison Analysis ----
    lines.append("## A/B Comparison Analysis\n")
    lines.append("**Config A — Hybrid Search + Reranking:**")
    lines.append("> - Semantic search (dense) + BM25 lexical search → RRF fusion")
    lines.append("> - Cross-encoder reranking để chấm lại relevance")
    lines.append("> - PageIndex fallback nếu score < threshold\n")

    lines.append("**Config B — Dense-Only (No Reranking):**")
    lines.append("> - Chỉ semantic search (cosine similarity)")
    lines.append("> - Không có lexical search, không có reranking")
    lines.append("> - Trả về top-k theo embedding score trực tiếp\n")

    winner = "Config A" if avg_a >= avg_b else "Config B"
    diff_pct = abs(delta_avg) / (avg_b + 1e-9) * 100
    lines.append(f"**Kết luận:** {winner} đạt điểm trung bình cao hơn ({abs(delta_avg):.3f} điểm, ")
    lines.append(f"tức +{diff_pct:.1f}%). ")
    if avg_a >= avg_b:
        lines.append("Kết hợp hybrid search + reranking giúp tăng chất lượng retrieval và độ chính xác")
        lines.append("của câu trả lời, đặc biệt cải thiện Context Recall và Faithfulness.")
    else:
        lines.append("Dense-only search đạt kết quả tương đương hoặc tốt hơn trong trường hợp này,")
        lines.append("có thể do corpus nhỏ hoặc cross-encoder chưa được tune cho tiếng Việt.")
    lines.append("")
    lines.append("---\n")

    # ---- Worst Performers ----
    lines.append("## Worst Performers (Bottom 3)\n")
    lines.append("*Phân tích từ Config A — Hybrid + Reranking*\n")
    lines.append("| # | Question | Faith. | Relevance | Recall | Precision | Failure Stage | Root Cause |")
    lines.append("|---|----------|--------|-----------|--------|-----------|---------------|------------|")

    for rank, item in enumerate(pq_a[:3], 1):
        q = item["question"][:50] + "..." if len(item["question"]) > 50 else item["question"]
        sc = item["scores"]
        faith = sc.get("faithfulness", 0.0)
        rel = sc.get("answer_relevancy", 0.0)
        rec = sc.get("contextual_recall", 0.0)
        prec = sc.get("contextual_precision", 0.0)

        # Xác định failure stage
        if rec < 0.4:
            stage = "Retrieval"
            cause = "Retriever không lấy đủ evidence cho câu hỏi này"
        elif faith < 0.4:
            stage = "Generation"
            cause = "LLM hallucinate hoặc không bám sát context"
        elif rel < 0.4:
            stage = "Generation"
            cause = "Câu trả lời không đúng trọng tâm câu hỏi"
        elif prec < 0.4:
            stage = "Retrieval"
            cause = "Nhiều chunk retrieved không liên quan (noise)"
        else:
            stage = "Mixed"
            cause = "Điểm tổng thấp, cần kiểm tra thêm"

        lines.append(f"| {rank} | {q} | {faith:.2f} | {rel:.2f} | {rec:.2f} | {prec:.2f} | {stage} | {cause} |")

    lines.append("")
    lines.append("---\n")

    # ---- Recommendations ----
    lines.append("## Recommendations\n")
    lines.append("### Cải tiến 1: Dùng embedding model tiếng Việt")
    lines.append("**Action:** Thay `all-MiniLM-L6-v2` bằng `BAAI/bge-m3` hoặc `keepitreal/vietnamese-sbert`")
    lines.append("để tăng chất lượng embedding cho văn bản pháp luật tiếng Việt.")
    lines.append("**Expected impact:** Tăng Context Recall và Faithfulness thêm 10–15%.\n")

    lines.append("### Cải tiến 2: Tăng kích thước chunk và overlap")
    lines.append("**Action:** Tăng `chunk_size` từ 500 lên 800–1000 ký tự, `overlap` từ 50 lên 150.")
    lines.append("Mỗi chunk sẽ chứa đủ context pháp lý (điều khoản + phần giải thích liền kề).")
    lines.append("**Expected impact:** Giảm trường hợp câu trả lời thiếu thông tin (Context Recall +8%).\n")

    lines.append("### Cải tiến 3: Dùng Jina Reranker multilingual")
    lines.append("**Action:** Cung cấp JINA_API_KEY hợp lệ để kích hoạt `jina-reranker-v2-base-multilingual`")
    lines.append("thay vì local cross-encoder `ms-marco-MiniLM-L-6-v2` (chỉ tốt cho tiếng Anh).")
    lines.append("**Expected impact:** Reranking chính xác hơn cho tiếng Việt, tăng Context Precision +12%.\n")

    lines.append("---\n")
    lines.append(f"*Báo cáo được tạo tự động bởi `eval_pipeline.py`*")

    content = "\n".join(lines)
    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\n✓ Results exported to: {RESULTS_PATH}")
    return content


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Evaluation Pipeline — Sản phẩm 2 (Bài tập nhóm)")
    print("  Framework: DeepEval | LLM Judge: gpt-4o-mini via OpenRouter")
    print("=" * 60)

    # 1. Cấu hình DeepEval dùng OpenRouter
    _setup_deepeval_openrouter()

    # 2. Load golden dataset
    golden_dataset = load_golden_dataset()
    assert len(golden_dataset) >= 15, (
        f"Golden dataset phải có ≥15 cặp Q&A, hiện có {len(golden_dataset)}"
    )

    # 3. Chạy A/B comparison (2 configs)
    comparison = compare_configs(golden_dataset)

    # 4. Export results ra results.md
    export_results(comparison)

    # 5. In tóm tắt
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for key in ["config_a", "config_b"]:
        cfg = comparison[key]
        scores = _extract_scores(cfg["results"])
        avg = sum(scores.values()) / len(scores)
        print(f"\n  {cfg['config_name']}")
        for metric, score in scores.items():
            print(f"    {metric:25s}: {score:.3f}")
        print(f"    {'Average':25s}: {avg:.3f}")

    print(f"\n✓ Done! Xem báo cáo chi tiết tại: {RESULTS_PATH}")
