"""
CrewAI agents for price comparison across Zepto, Blinkit, and Swiggy Instamart.
Uses Grok (xAI) as the LLM and DuckDuckGo for web search.
"""

import os
import ssl

# Bypass SSL verification for corporate proxies — must be set before imports
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("SSL_CERT_FILE", "")
ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001

from crewai import Agent, Task, Crew, Process, LLM  # noqa: E402
from crewai.tools import BaseTool  # noqa: E402


def build_llm() -> LLM:
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY environment variable not set.")
    return LLM(
        model="openai/grok-3",
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web for current product prices on quick-commerce platforms in India. "
        "Input: a search query string."
    )

    def _run(self, query: str) -> str:
        from ddgs import DDGS  # lazy import to avoid startup SSL errors
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No results found for this query."
            return "\n".join(
                f"{r['title']}: {r['body']} (source: {r['href']})" for r in results
            )
        except Exception as exc:
            return f"Search failed: {exc}"


def compare_prices(product: str) -> str:
    """Run the CrewAI pipeline and return a price comparison report."""
    llm = build_llm()
    search_tool = WebSearchTool()

    # Agent 1: researches live prices per platform
    price_researcher = Agent(
        role="Price Researcher",
        goal=(
            "Find the current prices for a product on Zepto, Blinkit, "
            "and Swiggy Instamart by searching the web."
        ),
        backstory=(
            "You are an expert at finding real-time prices from Indian "
            "quick-commerce apps. You search each platform individually "
            "to gather accurate price data."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        max_iter=6,
    )

    # Agent 2: analyses the data and writes the comparison report
    price_analyst = Agent(
        role="Price Analyst",
        goal="Compare prices across platforms and clearly recommend the best deal.",
        backstory=(
            "You are a sharp analyst who turns raw price data into clean, "
            "easy-to-read comparison tables with actionable recommendations."
        ),
        llm=llm,
        verbose=True,
    )

    research_task = Task(
        description=(
            f'Search for the current price of "{product}" on each platform separately:\n'
            f'  1. "{product} price Zepto"\n'
            f'  2. "{product} price Blinkit"\n'
            f'  3. "{product} price Swiggy Instamart"\n'
            "Collect all prices, quantities, and any offers mentioned."
        ),
        expected_output=(
            "A detailed summary of prices found for each platform, "
            "including product name, price (₹), quantity/unit, and any offers."
        ),
        agent=price_researcher,
    )

    analysis_task = Task(
        description=(
            f'Using the price research, create a comparison report for "{product}".\n\n'
            "Format your response as:\n"
            "1. A markdown table: | Platform | Price | Quantity/Unit | Notes |\n"
            "2. A clear recommendation line: **Best deal: [Platform] at ₹[price]**\n"
            "3. Any relevant notes (discounts, delivery fee, offers).\n"
            "If a platform price is unavailable, mark it as N/A."
        ),
        expected_output=(
            "A formatted markdown price comparison table followed by a best-deal "
            "recommendation and relevant notes."
        ),
        agent=price_analyst,
        context=[research_task],
    )

    crew = Crew(
        agents=[price_researcher, price_analyst],
        tasks=[research_task, analysis_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)
