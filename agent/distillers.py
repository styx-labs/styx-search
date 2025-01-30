from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from services.azure_openai import llm
from agent.types.types import JobDescriptionDistillOutput, DistillSourceOutput
from agent.prompts import distill_job_description_prompt, distill_source_prompt


@traceable(name="distill_job_description")
def distill_job_description(
    raw_content: str, role_query: str
) -> JobDescriptionDistillOutput:
    """Extract skills, requirements and summary from a job description."""
    structured_llm = llm.with_structured_output(JobDescriptionDistillOutput)
    output = structured_llm.invoke(
        [
            SystemMessage(
                content=distill_job_description_prompt.format(
                    raw_content=raw_content, role_query=role_query
                )
            )
        ]
        + [HumanMessage(content="Extract the key information:")]
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
        + [HumanMessage(content="Extract key professional information:")]
    )
    return output


@traceable(name="distill_source")
def distill_source(
    raw_content: str,
    is_job_description: bool,
    candidate_full_name: str = None,
    role_query: str = None,
) -> str:
    """Route to appropriate distiller based on source type and return formatted content."""
    if is_job_description:
        # For job descriptions, just return raw content to be processed later
        return raw_content
    else:
        if not candidate_full_name:
            return ""
        result = distill_human(raw_content, candidate_full_name)
        return result.distilled_source
