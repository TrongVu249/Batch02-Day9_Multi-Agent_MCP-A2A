"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


from pathlib import Path
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from .task4_chunking_indexing import EMBEDDING_MODEL

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

_model = None
_vector_store = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        vectorstore_path = STANDARDIZED_DIR.parent / "vector_store.pkl"
        if vectorstore_path.exists():
            with open(vectorstore_path, "rb") as f:
                _vector_store = pickle.load(f)
        else:
            _vector_store = []
    return _vector_store

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    model = get_model()
    chunks = get_vector_store()
    if not chunks:
        return []

    # Embed query
    query_embedding = model.encode(query, convert_to_numpy=True)
    query_norm = np.linalg.norm(query_embedding)

    results = []
    for chunk in chunks:
        emb = np.array(chunk["embedding"])
        emb_norm = np.linalg.norm(emb)
        
        if query_norm > 0 and emb_norm > 0:
            score = np.dot(emb, query_embedding) / (emb_norm * query_norm)
        else:
            score = 0.0
            
        results.append({
            "content": chunk["content"],
            "score": float(score),
            "metadata": chunk.get("metadata", {})
        })

    # Sort descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
