import os
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()


def search_product_online(query: str, max_results: int = 5) -> list[dict]:
    tool = TavilySearch(max_results=max_results, api_key=os.getenv("TAVILY_API_KEY"))
    response = tool.invoke(f"product {query} specifications features")
    results = []
    for r in (response if isinstance(response, list) else response.get("results", [])):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        })
    return results
