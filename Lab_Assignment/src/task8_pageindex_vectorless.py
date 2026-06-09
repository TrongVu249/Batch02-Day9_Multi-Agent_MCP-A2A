"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY.startswith("pi_"):
        raise ValueError("PAGEINDEX_API_KEY is not set or invalid in .env")
        
    from pageindex import PageIndexClient
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    # PageIndex submit_document uploads files. We can find files in data/landing/legal/ and upload them
    legal_dir = STANDARDIZED_DIR.parent / "landing" / "legal"
    uploaded_docs = []
    if legal_dir.exists():
        for file in legal_dir.iterdir():
            if file.is_file() and file.suffix.lower() in [".pdf", ".docx", ".doc"]:
                print(f"Uploading {file.name} to PageIndex...")
                try:
                    res = client.submit_document(file_path=str(file))
                    doc_id = res.get("doc_id")
                    if doc_id:
                        uploaded_docs.append({
                            "filename": file.name,
                            "doc_id": doc_id
                        })
                        print(f"  ✓ Uploaded {file.name} (ID: {doc_id})")
                except Exception as e:
                    print(f"  ⚠ Failed to upload {file.name}: {e}")
                    
    # Save the mapping to a local json file
    doc_mapping_path = STANDARDIZED_DIR.parent / "pageindex_docs.json"
    import json
    with open(doc_mapping_path, "w", encoding="utf-8") as f:
        json.dump(uploaded_docs, f, indent=4)


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY.startswith("pi_"):
        raise ValueError("PAGEINDEX_API_KEY is not set or invalid in .env")
        
    from pageindex import PageIndexClient
    import time
    
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    # Attempt to load doc_ids from our mapping file first
    doc_mapping_path = STANDARDIZED_DIR.parent / "pageindex_docs.json"
    doc_ids = []
    
    if doc_mapping_path.exists():
        try:
            import json
            with open(doc_mapping_path, "r", encoding="utf-8") as f:
                uploaded_docs = json.load(f)
                doc_ids = [d["doc_id"] for d in uploaded_docs if "doc_id" in d]
        except Exception as e:
            print(f"Error reading doc mapping: {e}")
            
    # If not found in mapping, try to list them
    if not doc_ids:
        try:
            docs_res = client.list_documents(limit=50)
            doc_ids = [d["id"] for d in docs_res.get("documents", []) if "id" in d]
        except Exception as e:
            print(f"Error listing documents from PageIndex: {e}")
            
    if not doc_ids:
        return []
        
    results = []
    for doc_id in doc_ids:
        try:
            res = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = res.get("retrieval_id")
            if not retrieval_id:
                continue
                
            # Poll status
            for _ in range(20):  # Wait up to 10 seconds per document
                status_res = client.get_retrieval(retrieval_id)
                status = status_res.get("status")
                if status == "completed":
                    retrieved_items = status_res.get("results") or status_res.get("chunks") or status_res.get("data") or []
                    for r in retrieved_items:
                        text = ""
                        score = 1.0
                        metadata = {}
                        
                        if isinstance(r, dict):
                            text = r.get("text") or r.get("content") or r.get("snippet") or ""
                            score = r.get("score") or r.get("relevance_score") or 1.0
                            metadata = r.get("metadata") or {}
                        elif hasattr(r, "text"):
                            text = getattr(r, "text", "")
                            score = getattr(r, "score", 1.0)
                            metadata = getattr(r, "metadata", {})
                            
                        if text:
                            results.append({
                                "content": text,
                                "score": float(score),
                                "metadata": metadata,
                                "source": "pageindex"
                            })
                    break
                elif status == "failed":
                    break
                time.sleep(0.5)
        except Exception as e:
            print(f"Error querying pageindex doc {doc_id}: {e}")
            
    # Sort descending by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
