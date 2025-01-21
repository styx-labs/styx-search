from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from services.azure_openai import llm
from agent.types import QueriesOutput, SearchQuery
from agent.llm_functions import identify_roles
from agent.prompts import search_query_prompt


def normalize_search_results(search_response) -> list:
    """Convert different search response formats into a unified list of results."""
    if isinstance(search_response, dict):
        return search_response["results"]
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict):
                query = response.get("query", "")
                is_job_description = query.lower().endswith("job description")

                if "results" in response:
                    # Add query info to each result
                    for result in response["results"]:
                        result["query"] = query
                        result["is_job_description"] = is_job_description
                    sources_list.extend(response["results"])
                else:
                    # Handle single result case
                    response["query"] = query
                    response["is_job_description"] = is_job_description
                    sources_list.append(response)
            else:
                sources_list.extend(response)
        return sources_list
    return []


def deduplicate_and_format_sources(search_response) -> dict:
    """Process search results and return formatted sources with citations."""
    # Get unified list of results
    sources_list = normalize_search_results(search_response)

    # Deduplicate by URL
    unique_sources = {source["url"]: source for source in sources_list}

    return unique_sources


@traceable(name="get_search_queries")
def get_search_queries(
    candidate_full_name: str,
    candidate_context: str,
    job_description: str,
    number_of_queries: int,
) -> List[SearchQuery]:
    # First identify roles
    roles_output = identify_roles(candidate_context)
    # Generate role-specific queries
    role_queries = [
        SearchQuery(
            search_query=f"{role.company} {role.role}{' ' + role.team if role.team else ''} job description",
            is_job_description_query=True,
        )
        for role in roles_output.roles
    ]

    # Get general queries from LLM
    structured_llm = llm.with_structured_output(QueriesOutput)
    output = structured_llm.invoke(
        [
            SystemMessage(
                content=search_query_prompt.format(
                    candidate_full_name=candidate_full_name,
                    candidate_context=candidate_context,
                    job_description=job_description,
                    number_of_queries=number_of_queries,
                )
            )
        ]
        + [HumanMessage(content="Generate search queries.")]
    )

    # Add a query for the candidate's name
    output.queries.append(
        SearchQuery(search_query=candidate_full_name, is_job_description_query=False)
    )

    # Combine role queries with general queries
    output.queries = role_queries + output.queries
    return output
