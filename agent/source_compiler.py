from typing import List, Dict, Tuple
from models.linkedin import LinkedInProfile, AILinkedinJobDescription
from agent.distillers import distill_job_description


def separate_sources_by_type(sources: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Separate sources into job descriptions and other sources."""
    job_description_sources = []
    other_sources = []

    for source in sources:
        if source["is_job_description"]:
            job_description_sources.append(source)
        else:
            other_sources.append(source)

    return job_description_sources, other_sources


def format_citations(sources: List[Dict]) -> Tuple[str, List[Dict]]:
    """Format non-job-description sources into citations."""
    formatted_text = "Sources:\n\n"
    citation_list = []

    for i, source in enumerate(sources, 1):
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

    return formatted_text, citation_list


def update_experience_with_job_descriptions(
    experience, job_description_sources: List[Dict], max_sources: int = 3
) -> None:
    """Update a single experience entry with relevant job descriptions."""
    # Find matching sources for this experience
    matching_sources = [
        source
        for source in job_description_sources
        if source["query"]
        .lower()
        .strip()
        .startswith(f"{experience.company.lower()} {experience.title.lower()}")
    ]

    if matching_sources:
        # Get top N sources by confidence
        top_sources = sorted(matching_sources, key=lambda x: x["weight"], reverse=True)[
            :max_sources
        ]

        # Combine raw content from all sources
        combined_raw_content = "\n\n".join(
            source["raw_content"] for source in top_sources
        )

        # Generate a coherent summary using all sources
        job_description = distill_job_description(
            combined_raw_content, f"{experience.company} {experience.title}"
        )

        experience.summarized_job_description = AILinkedinJobDescription(
            role_summary=job_description.role_summary,
            skills=job_description.skills,
            requirements=job_description.requirements,
            sources=[source["url"] for source in top_sources],
        )


def update_profile_with_job_descriptions(
    profile: LinkedInProfile, job_description_sources: List[Dict]
) -> LinkedInProfile:
    """Update all experiences in a profile with job descriptions."""
    for experience in profile.experiences:
        update_experience_with_job_descriptions(experience, job_description_sources)
    return profile
