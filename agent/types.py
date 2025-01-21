from typing import List, Annotated, Optional
from typing_extensions import TypedDict
import operator
from pydantic import BaseModel, Field
from enum import Enum
from datetime import date


class TraitType(str, Enum):
    BOOLEAN = "BOOLEAN"
    SCORE = "SCORE"

    @classmethod
    def _missing_(cls, value: str):
        # Handle uppercase values by converting to lowercase
        if isinstance(value, str):
            return cls(value.upper())
        return None


class KeyTrait(BaseModel):
    trait: str
    description: str
    trait_type: TraitType
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


class AILinkedinJobDescription(BaseModel):
    role_summary: str
    skills: List[str]
    requirements: List[str]
    sources: List[str]


class LinkedInExperience(BaseModel):
    title: str
    company: str
    description: Optional[str] = None
    starts_at: Optional[date] = None
    ends_at: Optional[date] = None
    location: Optional[str] = None
    summarized_job_description: Optional[AILinkedinJobDescription] = None


class LinkedInEducation(BaseModel):
    school: str
    degree_name: Optional[str] = None
    field_of_study: Optional[str] = None
    starts_at: Optional[date] = None
    ends_at: Optional[date] = None


class LinkedInProfile(BaseModel):
    full_name: str
    occupation: Optional[str] = None
    headline: Optional[str] = None
    summary: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    public_identifier: str
    experiences: List[LinkedInExperience] = []
    education: List[LinkedInEducation] = []

    def to_context_string(self) -> str:
        """Convert the profile to a formatted string context."""
        context = ""
        if self.occupation:
            context += f"Current Occupation: {self.occupation}\n"
        if self.headline:
            context += f"Headline: {self.headline}\n"
        if self.summary:
            context += f"Summary: {self.summary}\n"
        if self.city:
            context += f"City: {self.city}\n"

        for exp in self.experiences:
            context += f"Experience: {exp.title} at {exp.company}"
            if exp.description:
                context += f" - {exp.description}"
            context += "\n"

        for edu in self.education:
            context += (
                f"Education: {edu.school}; {edu.degree_name} in {edu.field_of_study}\n"
            )

        return context

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            full_name=data["full_name"],
            occupation=data["occupation"],
            headline=data["headline"],
            summary=data["summary"],
            city=data["city"],
            country=data["country"],
            public_identifier=data["public_identifier"],
            experiences=data["experiences"],
            education=data["education"],
        )


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


class SearchInputState(TypedDict):
    job_description: str
    candidate_context: str
    candidate_profile: LinkedInProfile
    candidate_full_name: str
    key_traits: list[KeyTrait]
    number_of_queries: int
    confidence_threshold: float


class OutputState(TypedDict):
    citations: list[dict]
    sections: list[dict]
    summary: str
    overall_score: float
    source_str: str
    candidate_profile: LinkedInProfile


class Role(BaseModel):
    company: str
    role: str
    team: Optional[str] = None


class RolesOutput(BaseModel):
    roles: list[Role]
