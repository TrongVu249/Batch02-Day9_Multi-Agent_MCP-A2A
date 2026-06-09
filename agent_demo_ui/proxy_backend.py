import asyncio
import os
import sys
import time
from uuid import uuid4
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path to load common modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from a2a.client import A2AClient
from a2a.types import (
    AgentCard,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    TextPart,
)
from common.a2a_client import _extract_text

app = FastAPI(title="Multi-Agent A2A UI Proxy Backend")

# Enable CORS for the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:10000")
CUSTOMER_AGENT_URL = os.getenv("CUSTOMER_AGENT_URL", "http://localhost:10100")

class QueryRequest(BaseModel):
    question: str

@app.get("/api/agents")
async def get_agents():
    """Proxy to list all registered agents from the registry."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{REGISTRY_URL}/agents")
            if resp.status_code == 200:
                return resp.json()
            else:
                return {"agents": [], "error": f"Registry returned status {resp.status_code}"}
        except Exception as e:
            return {"agents": [], "error": f"Could not connect to Registry: {str(e)}"}

@app.post("/api/query")
async def execute_query(payload: QueryRequest):
    """Bridges the A2A SDK client message routing for a user question."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=300.0) as http_client:
            # Resolve customer agent card
            card_url = f"{CUSTOMER_AGENT_URL}/.well-known/agent.json"
            try:
                card_resp = await http_client.get(card_url)
                card_resp.raise_for_status()
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Customer Agent at {CUSTOMER_AGENT_URL} is unreachable. Check if agents are running. Error: {str(e)}"
                )

            agent_card = AgentCard.model_validate(card_resp.json())
            client = A2AClient(httpx_client=http_client, agent_card=agent_card)

            # Construct A2A message request
            message = Message(
                role=Role.user,
                parts=[Part(root=TextPart(text=question))],
                message_id=str(uuid4()),
            )
            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(message=message),
            )

            # Route message and get response
            response = await client.send_message(request)
            elapsed = time.time() - start_time
            text_response = _extract_text(response)

            return {
                "status": "success",
                "answer": text_response,
                "elapsed_seconds": round(elapsed, 2),
                "trace_id": message.metadata.get("trace_id", "N/A") if message.metadata else "N/A"
            }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "status": "error",
            "detail": f"Error communicating with agents: {str(e)}",
            "elapsed_seconds": round(elapsed, 2)
        }

if __name__ == "__main__":
    print("Starting proxy backend on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
