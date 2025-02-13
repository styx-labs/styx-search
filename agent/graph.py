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
    EvaluationInputState,
)
from services.tavily import tavily_search_async
from langserve import RemoteRunnable
import os


def generate_queries(state: SearchState):
    content = get_search_queries(
        state.job.job_description,
        state.number_of_queries,
        state.profile,
    )

    return {"search_queries": content.queries}


async def gather_sources(state: SearchState):
    all_sources = await tavily_search_async(state.search_queries)
    unvalidated_sources = deduplicate_and_format_sources(all_sources)
    return {"unvalidated_sources": unvalidated_sources}


def initiate_source_validation(state: SearchState):
    return [
        Send("validate_and_distill_source", state.model_copy(update={"source": source}))
        for source in state.unvalidated_sources.keys()
    ]


def validate_and_distill_source(state: SearchState):
    source = state.unvalidated_sources[state.source]
    if source["raw_content"] is None:
        return {"validated_sources": []}

    source["raw_content"] = trim_text(source["raw_content"])

    confidence = validate_source(
        raw_content=source["raw_content"],
        title=source["title"],
        candidate_full_name=state.profile.full_name,
        candidate_context=state.profile.to_context_string(),
        role_query=source["query"],
        is_job_description=source["is_job_description"],
    )

    if confidence < state.confidence_threshold:
        return {"validated_sources": []}

    source["weight"] = confidence
    source["distilled_content"] = distill_source(
        raw_content=source["raw_content"],
        is_job_description=source["is_job_description"],
        candidate_full_name=state.profile.full_name,
        role_query=source["query"],
    )

    return {"validated_sources": [source]}


def compile_sources(state: SearchState):
    ranked_sources = sorted(
        state.validated_sources, key=lambda x: x["weight"], reverse=True
    )

    job_description_sources, other_sources = separate_sources_by_type(ranked_sources)

    source_str, citations = format_citations(other_sources)

    profile = update_profile_with_job_descriptions(
        state.profile, job_description_sources
    )

    return {
        "source_str": source_str,
        "citations": citations,
        "profile": profile,
    }


async def get_evaluation(state: SearchState):
    remote_eval = RemoteRunnable(os.getenv("EVAL_ENDPOINT"))
    evaluation = await remote_eval.ainvoke(
        input=EvaluationInputState(
            source_str=state.source_str,
            candidate_profile=state.profile,
            job=state.job,
            citations=state.citations,
            custom_instructions=state.custom_instructions,
        )
    )
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
