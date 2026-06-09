# Lab 9 — Multi-Agent System với A2A Protocol: Tổng Kết

**Ngày học:** 09/06/2026  
**Codelab:** Xây Dựng Hệ Thống Multi-Agent với A2A Protocol  
**Công nghệ:** Python 3.11+, LangGraph, LangChain, A2A SDK, OpenRouter  

---

## Mục Tiêu Đạt Được

| # | Mục tiêu | Kết quả |
|---|-----------|---------|
| 1 | Hiểu cách LLM hoạt động từ cơ bản đến nâng cao | ✅ |
| 2 | Tích hợp tools và RAG vào LLM | ✅ |
| 3 | Xây dựng Single Agent với ReAct pattern | ✅ |
| 4 | Tạo Multi-Agent System với LangGraph | ✅ |
| 5 | Triển khai Distributed Agents với A2A Protocol | ✅ |
| 6 | Bài tập cộng điểm: Demo UI + Tối ưu hóa latency | ✅ |
| 7 | Bài tập nâng cao: Supervisor-Workers RAG System | ✅ |

---

## Phần 1: Direct LLM Calling

### Lý Thuyết
LLM ở dạng cơ bản nhất là API nhận input text và trả về output text — không có memory, không có tools, chỉ dựa vào training data.

### Việc Đã Làm
- **Bài Tập 1.2:** Thêm `temperature=0.3` vào `common/llm.py` để ổn định output.
- Kết quả: `get_llm()` cache singleton instance, tránh khởi tạo lại kết nối HTTP mỗi lần gọi.

```python
# common/llm.py — Singleton LLM instance với temperature control
_llm_instance = None

def get_llm() -> ChatOpenAI:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
            temperature=0.3,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
        )
    return _llm_instance
```

---

## Phần 2: LLM + RAG & Tools

### Lý Thuyết
- **RAG:** LLM tra cứu knowledge base trước khi trả lời.
- **Tools:** Function mà LLM gọi để thực hiện tác vụ cụ thể.
- **Function Calling Flow:** LLM → quyết định tool → execute → nhận kết quả → tổng hợp.

### Việc Đã Làm
- **Bài Tập 2.1:** Thêm entry `labor_law` vào `LEGAL_KNOWLEDGE` (Bộ luật Lao động VN 2019).
- **Bài Tập 2.2:** Tạo tool `check_statute_of_limitations(case_type)` trả về thời hiệu khởi kiện.

---

## Phần 3: Single Agent với ReAct Pattern

### Lý Thuyết
**ReAct = Reasoning + Acting:** Agent tự lặp chu trình Think → Act → Observe cho đến khi có câu trả lời.

### Việc Đã Làm
- **Bài Tập 3.1:** Thêm tool `search_case_law(keywords)` tra cứu án lệ.
- **Bài Tập 3.2:** Thêm `verbose=True` vào `create_react_agent()` để debug reasoning.

---

## Phần 4: Multi-Agent In-Process với LangGraph

### Lý Thuyết
**LangGraph StateGraph:** Định nghĩa state dùng chung, tạo nodes (bước xử lý), định nghĩa edges (luồng điều khiển). **Send API** cho phép dispatch nhiều tasks song song.

### Việc Đã Làm — `exercises/exercise_4_multiagent.py`

**Bài Tập 4.1:** Thêm `privacy_agent` — chuyên gia GDPR và bảo vệ dữ liệu cá nhân.

**Bài Tập 4.2:** Implement conditional routing trong `check_routing()`:

```python
def check_routing(state: State) -> list[Send]:
    question_lower = state["question"].lower()
    tasks = []
    if any(kw in question_lower for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))
    if any(kw in question_lower for kw in ["compliance", "sec", "regulation"]):
        tasks.append(Send("compliance_agent", state))
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
        tasks.append(Send("privacy_agent", state))
    return tasks if tasks else [Send("aggregate_results", state)]
```

**Kiến trúc Graph:**
```
START → law_agent → check_routing (conditional)
                         ├── tax_agent ──────────┐
                         ├── compliance_agent ───┼── aggregate_results → END
                         └── privacy_agent ──────┘
```

---

## Phần 5: Distributed A2A System

### Kiến Trúc Hệ Thống
```
Registry (port 10000) ← agents tự đăng ký khi khởi động
      ↓
Customer Agent (10100) → Law Agent (10101)
                                ↓
                   ┌────────────┴────────────┐
                   ↓                         ↓
         Tax Agent (10102)    Compliance Agent (10103)
```

### Khác Biệt so với Stage 4
| Tiêu chí | Stage 4 In-Process | Stage 5 Distributed |
|----------|-------------------|---------------------|
| Deploy | Single process | Mỗi agent = service độc lập |
| Giao tiếp | Function call | HTTP / A2A Protocol |
| Discovery | Hardcoded | Dynamic qua Registry |
| Scale | Tất cả cùng scale | Scale từng agent riêng |
| Fault isolation | Không có | Có thể xử lý lỗi từng agent |

### Bài Tập 5.3 — Sửa System Prompt Compliance Agent
Thêm giới hạn độ dài output vào `compliance_agent/graph.py`:

```python
COMPLIANCE_SYSTEM_PROMPT = """...
CRITICAL: Keep your response extremely brief, concise, and straight to the point.
Limit your entire response to under 150 words.
..."""
```

---

## Bài Tập Cộng Điểm

### 1. Agent Demo UI — `agent_demo_ui/index.html`

Xây dựng giao diện web hiển thị tương tác của các agents trong Stage 5 (Distributed A2A):

**Tính năng:**
- 🖥️ **SVG Topology Graph** — Visualize kiến trúc 5 nodes (Client → Customer → Law → Tax/Compliance) với animated data flow.
- 🟢 **Live Status Polling** — Tự động check registry mỗi 3 giây, hiển thị online/offline/working per node.
- 📋 **Simulate Mode** — Phát lại từng bước delegation pipeline với hiệu ứng animation, không cần agents chạy thực.
- 🔗 **Real A2A Mode** — Gọi thực tế tới Python agents qua `proxy_backend.py` (FastAPI CORS proxy).
- 🔍 **Agent Card Inspector** — Click vào node để xem system prompt, port, và skills.
- 📡 **Orchestration Log Terminal** — Hiển thị log theo thời gian thực với màu sắc phân loại.
- 💬 **Custom Query Input** — Nhập câu hỏi tuỳ chỉnh và 3 template question nhanh.

**Tech stack:** HTML + Vanilla CSS + ES Modules, Vite dev server, Google Fonts (Outfit + Fira Code).

### 2. Đo Lường Latency & Tối Ưu Hóa

#### Kết Quả Đo
| Trạng thái | Trước tối ưu | Sau tối ưu |
|-----------|-------------|-----------|
| Cold start (lần đầu) | **40.89 giây** | **17.08 giây** |
| Warm cache | **28.91 giây** | **9.93 giây** |
| **Cải thiện** | baseline | **~3-4x nhanh hơn** |

#### Phương Án Tối Ưu Áp Dụng

**① Singleton LLM Instance (`common/llm.py`)**  
Cache instance `ChatOpenAI`, giữ HTTP/TLS keep-alive tới OpenRouter, tránh bắt tay lại mỗi request.

```python
_llm_instance = None
def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(...)
    return _llm_instance
```

**② Registry Discovery Cache (`common/registry_client.py`)**  
Cache endpoint sau lần đầu discover, tránh gọi HTTP đến Registry mỗi request.

```python
_discover_cache: dict[str, str] = {}

async def discover(task: str) -> str:
    if task in _discover_cache:
        return _discover_cache[task]  # Cache hit — no HTTP call
    # ...fetch and store...
    _discover_cache[task] = endpoint
```

**③ Agent Card & httpx Client Cache (`common/a2a_client.py`)**  
Cache `AgentCard` sau lần đầu tải `agent.json`, dùng chung `httpx.AsyncClient` cho tất cả requests.

```python
_agent_card_cache: dict[str, AgentCard] = {}
_shared_client: httpx.AsyncClient | None = None

async def delegate(endpoint, ...):
    global _shared_client
    if _shared_client is None:
        _shared_client = httpx.AsyncClient(timeout=300.0)
    if endpoint not in _agent_card_cache:
        # Fetch and cache agent card
        _agent_card_cache[endpoint] = AgentCard.model_validate(...)
```

**④ Prompt Length Optimization**  
Giới hạn output dưới 150 từ trong system prompts của Compliance Agent, Law Agent, Aggregator — giảm số token LLM sinh ra (nguyên nhân chính gây latency).

**⑤ Direct Delegation trong Customer Agent**  
Loại bỏ ReAct loop tại Customer Agent, thay bằng `StateGraph` chỉ có 1 node `call_law_agent` — giảm 1 lượt gọi LLM.

```python
# customer_agent/graph.py — Direct delegation, không qua ReAct loop
graph = StateGraph(CustomerState)
graph.add_node("call_law_agent", call_law_agent)
graph.set_entry_point("call_law_agent")
graph.add_edge("call_law_agent", END)
```

---

## Bài Tập Nâng Cao: Supervisor-Workers RAG System

### Mục Tiêu
Cải tiến **RAG Pipeline Day08** (pháp luật ma tuý VN + tin tức nghệ sĩ) bằng cách áp dụng **Supervisor-Workers pattern** với LangGraph.

### Kiến Trúc — `Lab_Assignment/src/supervisor_rag.py`

```
User Query
    │
    ▼
SUPERVISOR (Rule-based Routing — 0ms overhead)
    │ Phân loại: legal / news / general
    │ Chọn mode: hybrid (Semantic+BM25) vs dense-only
    │ Dispatch qua LangGraph Send API
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
WORKER 1: Retrieval               WORKER 2: Legal Analysis
① Semantic Search                 • System prompt luật ma tuý VN
② BM25 Lexical Search             • Điều 247-259 BLHS 2015
③ RRF Merge                       • Phân tích < 150 từ
④ Cross-Encoder Rerank
⑤ PageIndex Fallback
    │                                  │
    └──────────────┬───────────────────┘
                   ▼
              AGGREGATE (gộp W1 & W2)
                   │
                   ▼
    WORKER 3: Citation & Synthesis
    • Reorder chunks (lost-in-middle prevention)
    • Format source labels
    • LLM generation với citation [Nguồn, Năm] bắt buộc
                   │
                   ▼
         Final Answer có Citation
```

### So Sánh Monolithic vs Supervisor-Workers

| Aspect | Trước (Monolithic) | Sau (Supervisor-Workers) |
|--------|-------------------|-----------------------------|
| Architecture | Single function call | Multi-agent StateGraph |
| Parallelism | Sequential | Worker 1 + 2 chạy song song |
| Routing | Hardcoded | Supervisor phân loại & route |
| Legal Context | Không có | Worker 2 cung cấp phân tích chuyên sâu |
| Observability | Không log | Log từng bước rõ ràng |
| Extensibility | Khó thêm mới | Thêm worker = thêm node |
| Search mode | Luôn hybrid | Supervisor chọn mode tối ưu |

### Cách Chạy
```bash
# Từ project root
.venv\Scripts\python.exe -m Lab_Assignment.main

# Với câu hỏi tuỳ chỉnh
.venv\Scripts\python.exe -m Lab_Assignment.main "Hình phạt tội mua bán ma tuý?"
```

---

## Tổng Kết: 5 Stages So Sánh

| Stage | Pattern | Use Case | Độ phức tạp |
|-------|---------|----------|------------|
| 1 | Direct LLM | Câu hỏi đơn giản, không cần tools | ⭐ |
| 2 | LLM + Tools | Cần tra cứu data hoặc tính toán | ⭐⭐ |
| 3 | ReAct Agent | Tự động orchestration, multi-step | ⭐⭐⭐ |
| 4 | Multi-Agent In-Process | Nhiều domains, parallel processing | ⭐⭐⭐⭐ |
| 5 | Distributed A2A | Production, scalable, fault-tolerant | ⭐⭐⭐⭐⭐ |

---

## Files Đã Tạo / Sửa Đổi

| File | Hành động | Nội dung |
|------|-----------|----------|
| `common/llm.py` | Sửa | Singleton pattern + `temperature=0.3` |
| `common/registry_client.py` | Sửa | Thêm `_discover_cache` |
| `common/a2a_client.py` | Sửa | Cache agent card + shared httpx client |
| `compliance_agent/graph.py` | Sửa | Giới hạn output ≤ 150 từ trong system prompt |
| `customer_agent/graph.py` | Sửa | Direct delegation StateGraph thay ReAct loop |
| `exercises/exercise_4_multiagent.py` | Tạo mới | Privacy agent + conditional routing |
| `agent_demo_ui/index.html` | Tạo mới | Web UI demo A2A Stage 5 |
| `agent_demo_ui/src/` | Tạo mới | JavaScript modules (main.js, agents.js, simulation.js, ui.js) |
| `agent_demo_ui/proxy_backend.py` | Tạo mới | FastAPI CORS proxy cho Real A2A Mode |
| `Lab_Assignment/src/supervisor_rag.py` | Tạo mới | Supervisor-Workers RAG System |
| `Lab_Assignment/main.py` | Tạo mới | Demo script 3 test queries |
| `CODELAB.md` | Sửa | Thêm phần trả lời bài tập cộng điểm |

---

## Câu Hỏi Ôn Tập — Đáp Án

**1. Khi nào nên dùng single agent thay vì multi-agent?**  
Dùng single agent khi task đơn giản, một domain, không cần parallel processing. Multi-agent khi cần specialization, parallel execution, hoặc fault isolation.

**2. Ưu điểm của A2A protocol so với gRPC hoặc REST thông thường?**  
A2A có chuẩn discovery (Registry), agent cards mô tả capabilities, và trace propagation built-in — giảm boilerplate khi xây distributed agent systems.

**3. Làm thế nào để prevent infinite delegation loops trong A2A?**  
Dùng `delegation_depth` counter trong metadata. Mỗi agent kiểm tra `depth < MAX_DELEGATION_DEPTH` trước khi delegate tiếp.

**4. Tại sao cần Registry service? Có thể hardcode URLs không?**  
Hardcode URLs không scale và không fault-tolerant. Registry cho phép dynamic discovery — khi agent restart ở port khác hoặc thêm agent mới, không cần sửa code.
