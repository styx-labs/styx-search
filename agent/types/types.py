from typing import List, Annotated, Optional
from typing_extensions import TypedDict
import operator
from pydantic import BaseModel, Field
from .linkedin import LinkedInProfile


class KeyTrait(BaseModel):
    trait: str
    description: str
    value_type: Optional[str] = None
    required: bool = True


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")
    is_job_description_query: bool = Field(
        False, description="Whether this query is for job descriptions."
    )


class QueriesOutput(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )


class ValidationOutput(BaseModel):
    confidence: float


class JobDescriptionValidationOutput(BaseModel):
    confidence: float


class DistillSourceOutput(BaseModel):
    distilled_source: str


class JobDescriptionDistillOutput(BaseModel):
    skills: list[str]
    requirements: list[str]
    role_summary: str


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


class Role(BaseModel):
    company: str
    role: str
    team: Optional[str] = None


class RolesOutput(BaseModel):
    roles: list[Role]
