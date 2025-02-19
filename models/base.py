from typing import Optional
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")
    is_job_description_query: bool = Field(
        False, description="Whether this query is for job descriptions."
    )


class QueriesOutput(BaseModel):
    queries: list[SearchQuery] = Field(
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


class Role(BaseModel):
    company: str
    role: str
    team: Optional[str] = None


class RolesOutput(BaseModel):
    roles: list[Role]
