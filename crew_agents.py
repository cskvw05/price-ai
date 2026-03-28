"""
Multi-agent price comparison pipeline using Grok (xAI) directly.
Implements the same Researcher → Analyst pattern as CrewAI, without the dependency.
"""

import json
import os
import ssl

import httpx
from openai import OpenAI

# SSL bypass for corporate proxies
ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
os.environ.setdefault("CURL_CA_BUNDLE", "")

SEARCH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current product prices on Indian quick-commerce platforms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"}
                },
                "required": ["query"],
            },
        },
    }
]


def _build_client() -> OpenAI:
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY environment variable not set.")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
        http_client=httpx.Client(verify=False),
    )


def _web_search(query: str) -> str:
    from ddgs import DDGS  # lazy import
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        return "\n".join(
            f"{r['title']}: {r['body']} (source: {r['href']})" for r in results
        )
    except Exception as exc:
        return f"Search failed: {exc}"


def _run_agent(client: OpenAI, system: str, user_msg: str, tools=None) -> str:
    """Single agent: call Grok with optional tool use and return the final text."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    while True:
        kwargs: dict = {"model": "grok-3", "max_tokens": 4096, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        # Build a serializable assistant entry for history
        entry: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(entry)

        if choice.finish_reason == "stop":
            return msg.content or ""

        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = _web_search(args.get("query", ""))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            return msg.content or ""


def compare_prices(product: str) -> str:
    """
    Two-agent pipeline:
      Agent 1 (Researcher) – searches live prices on each platform.
      Agent 2 (Analyst)    – compiles the data into a comparison table.
    """
    client = _build_client()

    # ── Agent 1: Price Researcher ────────────────────────────────────────────
    research = _run_agent(
        client,
        system=(
            "You are a price researcher for Indian quick-commerce apps. "
            "Use web_search to find prices on Zepto, Blinkit, and Swiggy Instamart "
            "separately. Summarise all prices, quantities, and offers you find."
        ),
        user_msg=(
            f'Find the current price of "{product}" on Zepto, Blinkit, and '
            "Swiggy Instamart. Search each platform with its own query."
        ),
        tools=SEARCH_TOOL,
    )

    # ── Agent 2: Price Analyst ───────────────────────────────────────────────
    analysis = _run_agent(
        client,
        system=(
            "You are a price analyst. Given research data, produce a clear "
            "markdown comparison table and a best-deal recommendation."
        ),
        user_msg=(
            f'Price research for "{product}":\n\n{research}\n\n'
            "Now write:\n"
            "1. A markdown table — columns: Platform | Price | Quantity/Unit | Notes\n"
            "2. **Best deal: [Platform] at ₹[price]**\n"
            "3. Any relevant notes (discounts, delivery fee, offers).\n"
            "Mark unavailable prices as N/A."
        ),
    )

    return analysis
