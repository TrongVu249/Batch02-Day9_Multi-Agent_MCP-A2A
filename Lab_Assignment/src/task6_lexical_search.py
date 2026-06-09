"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path
import pickle
from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# We will lazily load and cache the corpus and BM25 index
_bm25 = None
_corpus = None

def load_corpus_and_bm25():
    global _bm25, _corpus
    if _corpus is None:
        vectorstore_path = STANDARDIZED_DIR.parent / "vector_store.pkl"
        if vectorstore_path.exists():
            with open(vectorstore_path, "rb") as f:
                _corpus = pickle.load(f)
        else:
            _corpus = []
            
        if _corpus:
            tokenized_corpus = [doc["content"].lower().split() for doc in _corpus]
            _bm25 = BM25Okapi(tokenized_corpus)

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    if not corpus:
        return None
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    load_corpus_and_bm25()
    if not _corpus or not _bm25:
        return []
        
    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)
    
    results = []
    for idx, score in enumerate(scores):
        if score > 0:
            results.append({
                "content": _corpus[idx]["content"],
                "score": float(score),
                "metadata": _corpus[idx].get("metadata", {})
            })
            
    # Sort descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
