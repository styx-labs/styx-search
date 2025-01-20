import re
from typing import Optional
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from agent.azure_openai import llm
from langsmith import traceable
from agent.types import (
    QueriesOutput,
    ValidationOutput,
    JobDescriptionValidationOutput,
    DistillSourceOutput,
    JobDescriptionDistillOutput,
    SearchQuery,
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


def job_description_heuristic_validator(
    content: str, title: str, role_query: str
) -> bool:
    """Validate if content likely contains a job description based on the role query."""
    cleaned_link_text = clean_text(content + " " + title)
    cleaned_role_query = clean_text(role_query)
    query_parts = cleaned_role_query.split()

    score = 0.0

    # Check for job description indicators
    job_indicators = [
        "job",
        "position",
        "career",
        "opening",
        "opportunity",
        "role",
        "posting",
    ]
    if any(indicator in cleaned_link_text for indicator in job_indicators):
        score += 0.5

    # Check for role query matches
    query_part_matches = sum(
        1 for part in query_parts if f" {part} " in f" {cleaned_link_text} "
    )
    score += (query_part_matches / len(query_parts)) * 0.5

    return score >= 0.5


class Role(BaseModel):
    company: str
    role: str
    team: Optional[str] = None


class RolesOutput(BaseModel):
    roles: list[Role]


@traceable(name="identify_roles")
def identify_roles(candidate_context: str) -> RolesOutput:
    """Extract roles from candidate context."""
    structured_llm = llm.with_structured_output(RolesOutput)
    output = structured_llm.invoke(
        [
            SystemMessage(
                content="""Extract all professional roles from the given context.
                For each role, identify:
                - company
                - role/title
                - team/department (if available, usually a short name in the description identifies it)
                
                Return as a list of roles, where each role has:
                - company: string
                - role: string
                - team: string or null if not available"""
            ),
            HumanMessage(content=candidate_context),
        ]
    )
    return output


@traceable(name="get_search_queries")
def get_search_queries(
    candidate_full_name: str,
    candidate_context: str,
    job_description: str,
    number_of_queries: int,
) -> QueriesOutput:
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

    # Combine role queries with general queries
    output.queries = role_queries + output.queries
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


@traceable(name="job_description_llm_validator")
def job_description_llm_validator(
    raw_content: str, role_query: str
) -> JobDescriptionValidationOutput:
    """Validate if content contains a relevant job description using LLM."""
    structured_llm = llm.with_structured_output(JobDescriptionValidationOutput)
    output = structured_llm.invoke(
        [
            SystemMessage(
                content=f"""Given the following webpage content and a role query, determine if this content contains a relevant job description.
                
                Role query: {role_query}
                Webpage content: {raw_content}
                
                Consider:
                1. Does the content describe a job position or role?
                2. Does it match the role and company from the query?
                3. Is it a job description page rather than a news article, profile, or other content?
                
                Return a confidence score between 0 and 1."""
            )
        ]
        + [
            HumanMessage(
                content="Validate if this webpage contains a relevant job description."
            )
        ]
    )
    return output


@traceable(name="distill_human")
def distill_human(raw_content: str, candidate_full_name: str) -> DistillSourceOutput:
    """Extract relevant information about a person from raw content."""
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


@traceable(name="distill_job_description")
def distill_job_description(
    raw_content: str, role_query: str
) -> JobDescriptionDistillOutput:
    """Extract skills, requirements and summary from a job description."""
    structured_llm = llm.with_structured_output(JobDescriptionDistillOutput)
    output = structured_llm.invoke(
        [
            SystemMessage(
                content="""Given a job description, extract:
                1. A list of required or desired skills
                2. A list of other requirements (education, experience, etc.)
                3. A brief summary of the role (1-2 sentences)
                
                Raw content: {raw_content}
                Role query: {role_query}
                
                Focus on information that would help determine if someone has experience in this type of role.""".format(
                    raw_content=raw_content, role_query=role_query
                )
            )
        ]
        + [
            HumanMessage(
                content="Extract the key information from this job description."
            )
        ]
    )
    return output


@traceable(name="validate_source")
def validate_source(
    raw_content: str,
    title: str,
    candidate_full_name: str = None,
    candidate_context: str = None,
    role_query: str = None,
    is_job_description: bool = False,
) -> float:
    """Validate a source using both heuristic and LLM validators.
    Returns a confidence score between 0 and 1."""

    if is_job_description:
        if not role_query:
            raise ValueError("role_query is required for job description validation")

        # Run heuristic validator first
        if not job_description_heuristic_validator(raw_content, title, role_query):
            return 0.0

        # If passes heuristic, run LLM validator
        llm_result = job_description_llm_validator(raw_content, role_query)
        return llm_result.confidence
    else:
        if not candidate_full_name or not candidate_context:
            raise ValueError(
                "candidate_full_name and candidate_context are required for person validation"
            )

        # Run heuristic validator first
        if not heuristic_validator(raw_content, title, candidate_full_name):
            return 0.0

        # If passes heuristic, run LLM validator
        llm_result = llm_validator(raw_content, candidate_full_name, candidate_context)
        return llm_result.confidence


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


@traceable(name="distill_source")
def distill_source(
    raw_content: str,
    is_job_description: bool,
    candidate_full_name: str = None,
    role_query: str = None,
) -> str:
    """Route to appropriate distiller based on source type and return formatted content."""
    if is_job_description:
        if not role_query:
            raise ValueError("role_query is required for job description distillation")
        distilled = distill_job_description(raw_content, role_query)
        return f"Role Summary: {distilled.role_summary}\nSkills: {', '.join(distilled.skills)}\nRequirements: {', '.join(distilled.requirements)}"
    else:
        if not candidate_full_name:
            raise ValueError(
                "candidate_full_name is required for human source distillation"
            )
        return distill_human(raw_content, candidate_full_name).distilled_source
