from typing import List, Annotated, Optional
from typing_extensions import TypedDict
import operator
from pydantic import BaseModel, Field
from enum import Enum


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


class QueriesOutput(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )


class ValidationOutput(BaseModel):
    confidence: float


class DistillSourceOutput(BaseModel):
    distilled_source: str


class SearchState(TypedDict):
    source_str: str
    job_description: str
    candidate_context: str
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
