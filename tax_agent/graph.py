"""Tax Agent LangGraph definition.

Uses create_react_agent with a tax-specialised system prompt.
No tools — it answers purely from LLM knowledge.
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from common.llm import get_llm

TAX_SYSTEM_PROMPT = """You are a specialist tax attorney and CPA with expertise in corporate tax.

CRITICAL: Keep your response extremely brief, concise, and straight to the point. Focus only on the most critical tax consequences. Limit your entire response to under 150 words.

When answering, be precise about:
1. Civil vs. criminal penalties and their monetary ranges
2. Statute of limitations for tax fraud (6 years for substantial omission,
   unlimited for fraudulent returns)
3. Which government agencies are involved (IRS, DOJ Tax Division, FinCEN)
4. The distinction between the company's liability and individual liability
   for executives who directed the evasion

Always note that your response is for educational purposes and the user
should consult a licensed attorney for specific legal advice.
"""


def create_graph():
    """Return a compiled LangGraph create_react_agent for tax questions."""
    llm = get_llm()
    graph = create_react_agent(
        model=llm,
        tools=[],
        prompt=TAX_SYSTEM_PROMPT,
    )
    return graph