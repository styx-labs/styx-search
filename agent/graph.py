from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from agent.helper_functions import (
    deduplicate_and_format_sources,
    validate_source,
    distill_source,
    get_search_queries,
    distill_job_description,
)
from agent.types import (
    SearchState,
    SearchInputState,
    SearchQuery,
    OutputState,
    AILinkedinJobDescription,
    LinkedInProfile,
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

    confidence = validate_source(
        raw_content=source["raw_content"] if source["raw_content"] else "",
        title=source["title"],
        candidate_full_name=candidate_full_name,
        candidate_context=candidate_context,
        role_query=source["query"],
        is_job_description=source["is_job_description"],
    )

    if confidence < confidence_threshold:
        return {"validated_sources": []}

    source["weight"] = confidence
    source["distilled_content"] = distill_source(
        raw_content=source["raw_content"],
        is_job_description=source["is_job_description"],
        candidate_full_name=candidate_full_name,
        role_query=source["query"],
    )

    return {"validated_sources": [source]}


def compile_sources(state: SearchState):
    validated_sources = state["validated_sources"]
    ranked_sources = sorted(validated_sources, key=lambda x: x["weight"], reverse=True)

    # Separate job description sources from other sources
    job_description_sources = []
    other_sources = []

    for source in ranked_sources:
        if source["is_job_description"]:
            job_description_sources.append(source)
        else:
            other_sources.append(source)

    # Format non-job-description sources for citations
    formatted_text = "Sources:\n\n"
    citation_list = []

    for i, source in enumerate(other_sources, 1):
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

    # Convert candidate_profile dict to LinkedInProfile object
    candidate_profile = LinkedInProfile(**state["candidate_profile"])

    # Update LinkedIn Experience entries with job description sources
    for experience in candidate_profile.experiences:
        matching_sources = [
            source
            for source in job_description_sources
            if source["query"]
            .lower()
            .strip()
            .startswith(f"{experience.company.lower()} {experience.title.lower()}")
        ]
        if matching_sources:
            # Get top 3 sources by confidence
            top_sources = sorted(matching_sources, key=lambda x: x["weight"], reverse=True)[:3]
            
            # Combine raw content from all sources
            combined_raw_content = "\n\n".join(
                source["raw_content"] for source in top_sources
            )
            
            # Generate a coherent summary using all sources
            job_description = distill_job_description(
                combined_raw_content,
                f"{experience.company} {experience.title}"
            )
            
            experience.summarized_job_description = AILinkedinJobDescription(
                job_description=f"Role Summary: {job_description.role_summary}\nSkills: {', '.join(job_description.skills)}\nRequirements: {', '.join(job_description.requirements)}",
                sources=[source["url"] for source in top_sources],
            )

    # Convert back to dict for state
    candidate_profile_dict = candidate_profile.model_dump()

    return {
        "source_str": formatted_text.strip(),
        "citations": citation_list,
        "candidate_profile": candidate_profile_dict,
    }


def initiate_source_validation(state: SearchState):
    return [
        Send("validate_and_distill_source", {"source": source, **state})
        for source in state["sources_dict"].keys()
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
