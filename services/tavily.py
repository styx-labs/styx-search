from tavily import AsyncTavilyClient
import asyncio
from langsmith import traceable
from agent.get_secret import get_secret
import logging
from typing import Any
import random


tavily_async_client = AsyncTavilyClient(
    api_key=get_secret("tavily-api-key", "1")
)

async def exponential_backoff_retry(
    coroutine_func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exceptions=(Exception,)
) -> Any:
    """
    Executes a coroutine with exponential backoff retry logic.
    
    Args:
        coroutine_func: A function that returns a new coroutine
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exceptions: Tuple of exceptions to catch and retry on
    """
    for attempt in range(max_retries + 1):
        try:
            return await coroutine_func()
        except exceptions as e:
            if attempt == max_retries:
                raise e
            
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 0.1), max_delay)
            logging.warning(f"Attempt {attempt + 1} failed. Retrying in {delay:.2f} seconds... Error: {str(e)}")
            await asyncio.sleep(delay)

@traceable(name="tavily_search_async")
async def tavily_search_async(search_queries):
    """Performs concurrent web searches using the Tavily API."""
    search_tasks = []
    for query in search_queries:
        query_str = query.search_query
        # Wrap individual search in a traceable function
        search_tasks.append(_single_tavily_search(query_str))
    return await asyncio.gather(*search_tasks)

@traceable(name="single_tavily_search")
async def _single_tavily_search(query_str):
    """Performs a single web search using the Tavily API with retry logic."""
    return await exponential_backoff_retry(
        lambda: tavily_async_client.search(query_str, max_results=5, include_raw_content=True),
        max_retries=3,
        base_delay=1.0,
        max_delay=10.0
    )
