from datetime import datetime
from typing import Literal, Optional
from models.linkedin import LinkedInProfile
from pydantic import Field
from .serializable import SerializableModel


class KeyTrait(SerializableModel):
    trait: str
    description: str
    value_type: Optional[str] = None
    required: bool = True


class CalibratedProfiles(SerializableModel):
    """Represents a candidate to be calibrated"""

    url: str
    fit: Optional[Literal["good", "bad"]] = None
    reasoning: Optional[str] = None
    profile: Optional[LinkedInProfile] = None
    type: Literal["ideal", "pipeline"] = "pipeline"

    def __str__(self):
        output = ""
        if self.fit:
            output += f"Fit: {self.fit}\n"
        if self.reasoning:
            output += f"Reasoning: {self.reasoning}\n"
        if self.profile:
            output += f"Profile: {self.profile.to_context_string()}"
        return output.rstrip()


class Job(SerializableModel):
    """Represents a job posting"""

    job_description: str
    key_traits: list[KeyTrait]
    calibrated_profiles: list[CalibratedProfiles] = None
    job_title: str
    company_name: str
    created_at: datetime = Field(default_factory=datetime.now)
