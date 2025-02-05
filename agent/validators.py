from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from services.azure_openai import llm_4o, llm_4o_mini
from models.base import ValidationOutput, JobDescriptionValidationOutput
from agent.text_utils import clean_text
from agent.prompts import validate_job_description_prompt, validate_human_source_prompt


@traceable(name="job_description_heuristic_validator")
def job_description_heuristic_validator(
    content: str, title: str, role_query: str
) -> float:
    """Validate if content likely contains a job description based on the role query."""
    if not content or not role_query:
        return 0.0

    cleaned_link_text = clean_text(title)
    query_parts = clean_text(role_query).split()

    score = 0.0

    query_part_matches = sum(
        1 for part in query_parts if f" {part} " in f" {cleaned_link_text} "
    )
    score += query_part_matches / len(query_parts)

    return min(1.0, score)


@traceable(name="job_description_llm_validator")
def job_description_llm_validator(
    raw_content: str, role_query: str
) -> JobDescriptionValidationOutput:
    """Validate if content contains a relevant job description using LLM."""
    structured_llm = llm_4o.with_structured_output(JobDescriptionValidationOutput)
    output = structured_llm.invoke(
        [
            SystemMessage(
                content=validate_job_description_prompt.format(
                    raw_content=raw_content, role_query=role_query
                )
            ),
            HumanMessage(content="Generate a score"),
        ]
    )
    return output


@traceable(name="llm_validator")
def llm_validator(
    raw_content, candidate_full_name: str, candidate_context: str
) -> ValidationOutput:
    """Validate if content is about the candidate using LLM."""
    structured_llm = llm_4o_mini.with_structured_output(ValidationOutput)
    output = structured_llm.invoke(
        [
            SystemMessage(
                content=validate_human_source_prompt.format(
                    raw_content=raw_content,
                    candidate_full_name=candidate_full_name,
                    candidate_context=candidate_context,
                )
            ),
            HumanMessage(
                content="Rate how relevant this content is to the candidate (0-1)"
            ),
        ]
    )
    return output


def heuristic_validator(content, title, candidate_full_name: str) -> float:
    """Basic validation using text matching."""
    if not content or not candidate_full_name:
        return 0.0

    name_parts = candidate_full_name.lower().split()
    cleaned_title = clean_text(title)

    # Check if all parts of the name appear in the title
    if all(part in cleaned_title for part in name_parts):
        return 1.0
    return 0.0


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
            return 0.0

        # First check heuristic match
        heuristic_score = job_description_heuristic_validator(
            raw_content, title, role_query
        )
        if heuristic_score < 0.3:
            return 0.0

        # Then do detailed LLM validation
        llm_result = job_description_llm_validator(raw_content, role_query)
        return llm_result.confidence

    else:
        if not candidate_full_name or not candidate_context:
            return 0.0

        # First check heuristic match
        heuristic_score = heuristic_validator(raw_content, title, candidate_full_name)
        if heuristic_score < 0.3:
            return 0.0

        # Then do detailed LLM validation
        llm_result = llm_validator(raw_content, candidate_full_name, candidate_context)
        return llm_result.confidence
