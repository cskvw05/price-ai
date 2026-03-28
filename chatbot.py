"""
Price Comparison AI Chatbot
Compares prices from Zepto, Blinkit, and Swiggy Instamart using Grok + web search.
"""

import os
import json
import httpx
from openai import OpenAI
from duckduckgo_search import DDGS

XAI_BASE_URL = "https://api.x.ai/v1"
MODEL = "grok-3"

SYSTEM_PROMPT = """You are a smart price comparison assistant that helps users find the best deals
on grocery and daily essentials from quick-commerce platforms in India.

Your job:
1. When a user asks about a product's price, use the web_search tool to search for prices
   on Zepto, Blinkit, and Swiggy Instamart.
2. Search each platform specifically: e.g., "milk 1 litre price on Zepto",
   "milk 1 litre price Blinkit", "milk 1 litre price Swiggy Instamart"
3. Compile the results and present a clear comparison table.
4. Recommend the cheapest option (or best value considering quantity/offers).
5. If you can't find a price for a platform, mention it's unavailable or couldn't be fetched.

Always format your final response as:
- A price comparison table (platform | price | quantity/unit | notes)
- A clear recommendation: "Best deal: [Platform] at ₹[price]"
- Any relevant notes (discounts, offers, delivery time if mentioned)

Be conversational and helpful. Users may ask follow-up questions about specific products.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current product prices and availability on Zepto, Blinkit, and Swiggy Instamart.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g., 'Amul butter 500g price Blinkit'"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def create_client() -> OpenAI:
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY environment variable not set.")
    # Use a custom httpx client with SSL verification disabled (needed behind corporate proxies)
    http_client = httpx.Client(verify=False)
    return OpenAI(api_key=api_key, base_url=XAI_BASE_URL, http_client=http_client)


def web_search(query: str) -> str:
    """Execute a web search using DuckDuckGo and return results as text."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found for this query."
        return "\n".join(
            f"{r['title']}: {r['body']} (source: {r['href']})" for r in results
        )
    except Exception as e:
        return f"Search failed: {e}"


def run_agentic_loop(client: OpenAI, messages: list) -> str:
    """Run the Grok agentic loop, executing web searches until a final answer is ready."""
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=4096,
        )

        choice = response.choices[0]
        msg = choice.message

        # Build a serializable assistant message for history
        assistant_entry = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        # Model is done — return the final text
        if choice.finish_reason == "stop":
            return msg.content or ""

        # Model requested tool calls — execute each and feed results back
        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == "web_search":
                    args = json.loads(tc.function.arguments)
                    result = web_search(args["query"])
                else:
                    result = f"Unknown tool: {tc.function.name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            # Unexpected finish reason — return whatever text we have
            return msg.content or ""


def print_banner():
    print("=" * 60)
    print("  Price Comparison Chatbot  (powered by Grok)")
    print("  Zepto | Blinkit | Swiggy Instamart")
    print("=" * 60)
    print("Ask me to compare prices for any grocery/daily essential!")
    print("Examples:")
    print("  • Compare price of 1 litre milk")
    print("  • Which app has cheapest bread?")
    print("  • Price of Amul butter 500g")
    print('Type "exit" or "quit" to stop.\n')


def main():
    client = create_client()
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    print_banner()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye"):
            print("Goodbye! Happy shopping!")
            break

        conversation_history.append({"role": "user", "content": user_input})
        print("\nSearching prices... (this may take a moment)\n")

        try:
            reply = run_agentic_loop(client, conversation_history)
            print(f"Assistant: {reply}\n")
            print("-" * 60)
        except Exception as e:
            print(f"Error: {e}\n")
            # Remove the failed user message so history stays clean
            conversation_history.pop()


if __name__ == "__main__":
    main()
