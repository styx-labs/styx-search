import re
from langchain_core.messages import HumanMessage, SystemMessage
from agent.azure_openai import llm
from langsmith import traceable
from agent.types import (
    QueriesOutput,
    ValidationOutput,
    DistillSourceOutput,
)
from agent.prompts import (
    validation_prompt,
    distill_source_prompt,
    search_query_prompt,
)


def clean_text(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text.lower())


def heuristic_validator(content, title, candidate_full_name: str) -> bool:
    cleaned_link_text = clean_text(content + " " + title)
    cleaned_candidate_full_name = clean_text(candidate_full_name)
    name_parts = cleaned_candidate_full_name.split()

    score = 0.0

    if cleaned_candidate_full_name in cleaned_link_text:
        score += 1.0

    name_part_matches = sum(
        1 for part in name_parts if f" {part} " in f" {cleaned_link_text} "
    )
    score += (name_part_matches / len(name_parts)) * 0.5

    return score >= 0.5


@traceable(name="get_search_queries")
def get_search_queries(
    candidate_full_name: str,
    candidate_context: str,
    job_description: str,
    number_of_queries: int,
) -> QueriesOutput:
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
    return output


@traceable(name="llm_validator")
def llm_validator(
    raw_content, candidate_full_name: str, candidate_context: str
) -> ValidationOutput:
    structured_llm = llm.with_structured_output(ValidationOutput)
    output = structured_llm.invoke(
        [
            SystemMessage(
                content=validation_prompt.format(
                    candidate_full_name=candidate_full_name,
                    candidate_context=candidate_context,
                    raw_content=raw_content,
                )
            )
        ]
        + [
            HumanMessage(
                content="Validate if this webpage is about the candidate in question."
            )
        ]
    )
    return output


@traceable(name="distill_source")
def distill_source(raw_content, candidate_full_name: str) -> DistillSourceOutput:
    structured_llm = llm.with_structured_output(DistillSourceOutput)
    output = structured_llm.invoke(
        [
            SystemMessage(
                content=distill_source_prompt.format(
                    raw_content=raw_content, candidate_full_name=candidate_full_name
                )
            )
        ]
        + [
            HumanMessage(
                content="Extract the relevant information about the given person from the raw HTML."
            )
        ]
    )
    return output


def normalize_search_results(search_response) -> list:
    """Convert different search response formats into a unified list of results."""
    if isinstance(search_response, dict):
        return search_response["results"]
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and "results" in response:
                sources_list.extend(response["results"])
            else:
                sources_list.extend(response)
        return sources_list
    raise ValueError(
        "Input must be either a dict with 'results' or a list of search results"
    )


def deduplicate_and_format_sources(search_response) -> dict:
    """Process search results and return formatted sources with citations."""
    # Get unified list of results
    sources_list = normalize_search_results(search_response)

    # Deduplicate by URL
    unique_sources = {source["url"]: source for source in sources_list}

    return unique_sources
