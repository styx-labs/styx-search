from typing import List, Annotated, Optional
from typing_extensions import TypedDict
import operator
from .linkedin import LinkedInProfile
from .base import KeyTrait, SearchQuery


class SearchState(TypedDict):
    source_str: str
    job_description: str
    candidate_context: str
    candidate_profile: LinkedInProfile
    candidate_full_name: str
    key_traits: List[KeyTrait]
    number_of_queries: int
    search_queries: list[SearchQuery]
    validated_sources: Annotated[
        list, operator.add
    ]  # This is for parallelizing source validation
    citations: list[dict]
    confidence_threshold: float
    ideal_profiles: list[str]
    custom_instructions: Optional[str] = None


class SearchInputState(TypedDict):
    job_description: str
    candidate_context: str
    candidate_profile: LinkedInProfile
    candidate_full_name: str
    key_traits: list[KeyTrait]
    number_of_queries: int
    confidence_threshold: float
    ideal_profiles: list[str]
    custom_instructions: Optional[str] = None


class OutputState(TypedDict):
    citations: list[dict]
    sections: list[dict]
    summary: str
    required_met: int
    optional_met: int
    source_str: str
    candidate_profile: LinkedInProfile
    fit: int
    custom_instructions: Optional[str] = None
