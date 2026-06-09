"""Customer Agent LangGraph definition.

Uses create_react_agent with a `delegate_to_legal_agent` tool that:
1. Discovers the Law Agent via the registry
2. Sends the question to it via A2A
3. Returns the comprehensive legal response to the user

The tool accepts context propagation data (trace_id, context_id, depth)
via a closure — these are bound per-request in agent_executor.py.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from common.llm import get_llm

logger = logging.getLogger(__name__)

CUSTOMER_SYSTEM_PROMPT = """You are a helpful legal assistant at the front desk of a multi-agent
legal services platform. Your job is to:

1. Understand the user's legal question
2. Determine if it needs specialist legal analysis (contract issues, tax law,
   regulatory compliance, corporate liability, etc.)
3. If so, use the `delegate_to_legal_agent` tool to send it to the Law Agent,
   which will coordinate specialist sub-agents (Tax and Compliance) as needed
4. Present the comprehensive response clearly to the user

Always use the `delegate_to_legal_agent` tool for any substantive legal question.
Do not attempt to answer complex legal questions from your own knowledge alone.

Be professional, clear, and make the specialist response accessible to the user.
"""


from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage

class CustomerState(TypedDict):
    messages: list[BaseMessage]

def build_graph(trace_id: str, context_id: str, depth: int) -> Any:
    """Build a StateGraph that directly delegates the question to the Law Agent,
    bypassing customer-level LLM calls to reduce latency.
    """

    async def call_law_agent(state: CustomerState) -> dict:
        from common.a2a_client import delegate
        from common.registry_client import discover

        # Extract the question text
        question = ""
        for msg in reversed(state.get("messages", [])):
            if msg.content:
                question = str(msg.content)
                break

        logger.info(
            "Customer direct delegate to Law Agent | trace=%s context=%s depth=%d",
            trace_id, context_id, depth,
        )

        try:
            endpoint = await discover("legal_question")
            result = await delegate(
                endpoint=endpoint,
                question=question,
                context_id=context_id,
                trace_id=trace_id,
                depth=depth + 1,
            )
            if not result:
                result = "The Law Agent returned an empty response. Please try again."
            return {"messages": [AIMessage(content=result)]}
        except Exception as exc:
            logger.exception("Customer direct delegate failed: %s", exc)
            return {"messages": [AIMessage(content=f"Could not reach the Law Agent: {exc}")]}

    graph = StateGraph(CustomerState)
    graph.add_node("call_law_agent", call_law_agent)
    graph.set_entry_point("call_law_agent")
    graph.add_edge("call_law_agent", END)
    return graph.compile()