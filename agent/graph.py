from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from agent.distillers import distill_source
from agent.validators import (
    validate_source,
)
from agent.source_compiler import (
    separate_sources_by_type,
    format_citations,
    update_profile_with_job_descriptions,
    trim_text,
)
from agent.search import get_search_queries, deduplicate_and_format_sources
from models.search import (
    SearchState,
    SearchInputState,
    OutputState,
)
from models.linkedin import LinkedInProfile
from services.tavily import tavily_search_async
from langserve import RemoteRunnable
import os


def generate_queries(state: SearchState):
    content = get_search_queries(
        state["candidate_full_name"],
        state["candidate_context"],
        state["job_description"],
        state["number_of_queries"],
        state["candidate_profile"],
    )

    return {"search_queries": content.queries}


async def gather_sources(state: SearchState):
    all_sources = await tavily_search_async(state["search_queries"])
    unvalidated_sources = deduplicate_and_format_sources(all_sources)
    return {"unvalidated_sources": unvalidated_sources}


def validate_and_distill_source(state: SearchState):
    source = state["unvalidated_sources"][state["source"]]
    if source["raw_content"] is None:
        return {"validated_sources": []}
    
    source["raw_content"] = trim_text(source["raw_content"])

    confidence = validate_source(
        raw_content=source["raw_content"],
        title=source["title"],
        candidate_full_name=state["candidate_full_name"],
        candidate_context=state["candidate_context"],
        role_query=source["query"],
        is_job_description=source["is_job_description"],
    )

    if confidence < state["confidence_threshold"]:
        return {"validated_sources": []}

    source["weight"] = confidence
    source["distilled_content"] = distill_source(
        raw_content=source["raw_content"],
        is_job_description=source["is_job_description"],
        candidate_full_name=state["candidate_full_name"],
        role_query=source["query"],
    )

    return {"validated_sources": [source]}


def compile_sources(state: SearchState):
    validated_sources = state["validated_sources"]
    ranked_sources = sorted(validated_sources, key=lambda x: x["weight"], reverse=True)

    # Separate sources by type
    job_description_sources, other_sources = separate_sources_by_type(ranked_sources)

    # Format citations for non-job-description sources
    formatted_text, citation_list = format_citations(other_sources)

    # Convert candidate_profile dict to LinkedInProfile object
    candidate_profile = LinkedInProfile(**state["candidate_profile"])

    # Update profile with job descriptions
    updated_profile = update_profile_with_job_descriptions(
        candidate_profile, job_description_sources
    )

    # Convert back to dict for state
    candidate_profile_dict = updated_profile.model_dump()

    return {
        "source_str": formatted_text,
        "citations": citation_list,
        "candidate_profile": candidate_profile_dict,
    }


def initiate_source_validation(state: SearchState):
    return [
        Send("validate_and_distill_source", {"source": source, **state})
        for source in state["unvalidated_sources"].keys()
    ]


async def get_evaluation(state: SearchState):
    remote_eval = RemoteRunnable(os.getenv("EVAL_ENDPOINT"))
    evaluation = await remote_eval.ainvoke(input=state)
    return {**evaluation}


builder = StateGraph(SearchState, input=SearchInputState, output=OutputState)
builder.add_node("generate_queries", generate_queries)
builder.add_node("gather_sources", gather_sources)
builder.add_node("validate_and_distill_source", validate_and_distill_source)
builder.add_node("compile_sources", compile_sources)
builder.add_node("get_evaluation", get_evaluation)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "gather_sources")
builder.add_conditional_edges(
    "gather_sources", initiate_source_validation, ["validate_and_distill_source"]
)
builder.add_edge("validate_and_distill_source", "compile_sources")
builder.add_edge("compile_sources", "get_evaluation")
builder.add_edge("get_evaluation", END)

graph = builder.compile()
