from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from agent.helper_functions import (
    deduplicate_and_format_sources,
    heuristic_validator,
    llm_validator,
    distill_source,
    get_search_queries,
)
from agent.types import (
    SearchState,
    SearchInputState,
    SearchQuery,
    OutputState
)
from agent.tavily import tavily_search_async
from langserve import RemoteRunnable
import os


def generate_queries(state: SearchState):
    job_description = state["job_description"]
    candidate_context = state["candidate_context"]
    number_of_queries = state["number_of_queries"]
    candidate_full_name = state["candidate_full_name"]

    content = get_search_queries(
        candidate_full_name, candidate_context, job_description, number_of_queries
    )

    # Add a query for the candidate's name
    content.queries.append(SearchQuery(search_query=candidate_full_name))
    return {"search_queries": content.queries}


async def gather_sources(state: SearchState):
    search_docs = await tavily_search_async(state["search_queries"])
    sources_dict = deduplicate_and_format_sources(search_docs)
    return {"sources_dict": sources_dict}


def validate_and_distill_source(state: SearchState):
    source = state["sources_dict"][state["source"]]
    candidate_full_name = state["candidate_full_name"]
    candidate_context = state["candidate_context"]
    confidence_threshold = state["confidence_threshold"]

    if not heuristic_validator(
        source["raw_content"] if source["raw_content"] else "",
        source["title"],
        candidate_full_name,
    ):
        return {"validated_sources": []}

    llm_output = llm_validator(
        source["raw_content"], candidate_full_name, candidate_context
    )
    if llm_output.confidence < confidence_threshold:
        return {"validated_sources": []}

    source["weight"] = llm_output.confidence
    source["distilled_content"] = distill_source(
        source["raw_content"], candidate_full_name
    ).distilled_source
    return {"validated_sources": [source]}


def compile_sources(state: SearchState):
    validated_sources = state["validated_sources"]
    ranked_sources = sorted(validated_sources, key=lambda x: x["weight"], reverse=True)

    formatted_text = "Sources:\n\n"
    citation_list = []

    for i, source in enumerate(ranked_sources, 1):
        formatted_text += (
            f"[{i}]: {source['title']}:\n"
            f"URL: {source['url']}\n"
            f"Relevant content from source: {source['distilled_content']} "
            f"(Confidence: {source['weight']})\n===\n"
        )

        citation_list.append(
            {
                "index": i,
                "url": source["url"],
                "confidence": source["weight"],
                "distilled_content": source["distilled_content"],
            }
        )

    return {"source_str": formatted_text.strip(), "citations": citation_list}


def initiate_source_validation(state: SearchState):
    return [
        Send("validate_and_distill_source", {"source": source, **state})
        for source in state["sources_dict"].keys()
    ]


async def get_evaluation(state: SearchState):
    remote_eval = RemoteRunnable(os.getenv("EVAL_ENDPOINT"))
    evaluation = await remote_eval.ainvoke(input=state)
    return {**evaluation}


builder = StateGraph(
    SearchState, input=SearchInputState, output=OutputState
)
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
