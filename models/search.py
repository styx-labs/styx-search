from typing import Annotated, Optional
import operator
from .linkedin import LinkedInProfile
from .base import SearchQuery
from .jobs import Job
from .serializable import SerializableModel


class SearchState(SerializableModel):
    # Input
    profile: LinkedInProfile
    job: Job
    number_of_queries: int
    confidence_threshold: float = 0.8
    custom_instructions: Optional[str] = None

    # Intermediate
    search_queries: list[SearchQuery] = []
    unvalidated_sources: dict[str, dict] = {}
    validated_sources: Annotated[list, operator.add] = []

    # Output
    citations: list[dict] = []
    source_str: str = ""


class SearchInputState(SerializableModel):
    profile: LinkedInProfile
    job: Job
    number_of_queries: int
    confidence_threshold: float
    custom_instructions: Optional[str] = None


class EvaluationInputState(SerializableModel):
    source_str: str
    profile: LinkedInProfile
    job: Job
    citations: list[dict]
    custom_instructions: Optional[str] = None


class OutputState(SerializableModel):
    citations: list[dict]
    sections: list[dict]
    summary: str
    required_met: int
    optional_met: int
    source_str: str
    fit: int
    custom_instructions: Optional[str] = None
