import asyncio
from langsmith import traceable
import requests
from agent.get_secret import get_secret


url = "https://api.exa.ai/search"

@traceable(name="exa_search_async")
async def exa_search_async(queries):
    search_tasks = []
    for query in queries:
        search_tasks.append(_single_exa_search(query.search_query))
    return await asyncio.gather(*search_tasks)

@traceable(name="single_exa_search")
async def _single_exa_search(query):
    payload = {
        "query": query,
        "type": "keyword",
        "numResults": 10,
        "contents": {
            "text": {
                "maxCharacters": 400000,
                "includeHtmlTags": False
            },
            "livecrawl": "never",
        }
    }
    headers = {
        "x-api-key": get_secret("exa-api-key", "1"),
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    return response.json()
