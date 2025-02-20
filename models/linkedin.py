"""
LinkedIn data models with standardized serialization.
"""

from datetime import date
from .serializable import SerializableModel
from .career import CareerMetrics, FundingType


class Funding(SerializableModel):
    """Model for funding round data."""

    funding_type: FundingType = FundingType.UNKNOWN
    money_raised: int | None = None
    announced_date: date | None = None
    number_of_investors: int | None = None
    investor_list: list[str] = []


class LinkedInCompany(SerializableModel):
    """Model for LinkedIn company profile data."""

    company_id: str
    name: str
    website: str | None = None
    linkedin: str | None = None
    crunchbase: str | None = None
    location: dict[str, str] | None = None
    description: str | None = None
    industries: list[str] = []
    funding_data: list[Funding] = []
    founded_on: str | None = None
    ipo_status: str | None = None
    operating_status: str | None = None

    @property
    def funding_stage(self) -> FundingType:
        """Get the current funding stage of the company."""
        if not self.funding_data:
            return FundingType.UNKNOWN
        return self.funding_data[-1].funding_type

    def get_funding_stage_at_date(self, target_date: date) -> FundingType:
        """Get the company's funding stage at a specific date."""
        if not self.funding_data:
            return FundingType.UNKNOWN

        current_stage = FundingType.UNKNOWN

        for funding in sorted(
            [f for f in self.funding_data if f.announced_date],
            key=lambda x: x.announced_date,
        ):
            try:
                if funding.announced_date <= target_date:
                    current_stage = funding.funding_type
                else:
                    break
            except (ValueError, TypeError):
                continue

        return current_stage

    def get_funding_stages_between_dates(
        self, start_date: date, end_date: date = None, cutoff_date: date = None
    ) -> list[FundingType]:
        """Get the sequence of funding stages between two dates.

        Args:
            start_date: Start date of the period
            end_date: End date of the period (defaults to today)
            cutoff_date: Optional date before which to ignore funding rounds
        """
        if not self.funding_data:
            return []

        end_date = end_date or date.today()
        relevant_rounds = []

        # Filter funding data by cutoff date if provided
        valid_funding = [
            f
            for f in self.funding_data
            if f.announced_date and (not cutoff_date or f.announced_date) >= cutoff_date
        ]

        if not valid_funding:
            return [FundingType.UNKNOWN]

        # Get initial stage from valid funding rounds
        current_stage = self.get_funding_stage_at_date(start_date)

        for funding in sorted(valid_funding, key=lambda x: x.announced_date):
            try:
                funding_date = funding.announced_date
                if start_date < funding_date <= end_date:
                    stage = funding.funding_type
                    if stage != current_stage:
                        relevant_rounds.append(stage)
                        current_stage = stage
            except (ValueError, TypeError):
                continue

        return list(dict.fromkeys([current_stage] + relevant_rounds))

    def to_context_string(self) -> str:
        """Convert the company profile to a formatted string context."""
        context = f"Company: {self.name}\n\n"

        if self.description:
            context += f"Description: {self.description}\n\n"

        if self.location:
            location_str = ", ".join(
                filter(
                    None,
                    [
                        self.location.get("city"),
                        self.location.get("state"),
                        self.location.get("country"),
                    ],
                )
            )
            if location_str:
                context += f"Location: {location_str}\n\n"

        if self.industries:
            context += f"Industries: {', '.join(self.industries)}\n\n"

        if self.founded_on:
            context += f"Founded: {self.founded_on}\n\n"

        if self.funding_data:
            # Calculate total funding using proper attribute access
            total_funding = sum(
                round.money_raised
                for round in self.funding_data
                if round.money_raised is not None
            )
            if total_funding:
                context += f"Total Funding: ${total_funding:,.0f}\n"

            # Add latest funding round using proper date comparison
            latest_funding = next(
                (
                    round
                    for round in sorted(
                        self.funding_data,
                        key=lambda x: x.announced_date or date.min,
                        reverse=True,
                    )
                    if round.announced_date
                ),
                None,
            )

            if latest_funding:
                context += f"Latest Funding: {latest_funding.funding_type.value}"
                if latest_funding.money_raised is not None:
                    context += f" (${latest_funding.money_raised:,.0f})"
                if latest_funding.investor_list:
                    context += f"\nInvestors: {', '.join(latest_funding.investor_list)}"
                context += "\n\n"

        if self.ipo_status:
            context += f"IPO Status: {self.ipo_status}\n"

        if self.operating_status:
            context += f"Status: {self.operating_status}\n"

        return context.strip()


class AILinkedinJobDescription(SerializableModel):
    role_summary: str
    skills: list[str]
    requirements: list[str]
    sources: list[str]


class LinkedInExperience(SerializableModel):
    title: str | None
    company: str | None
    description: str | None
    starts_at: date | None
    ends_at: date | None
    location: str | None
    company_linkedin_profile_url: str | None
    company_data: LinkedInCompany | None = None
    summarized_job_description: AILinkedinJobDescription | None = None
    experience_tags: list[str] | None = None


class LinkedInEducation(SerializableModel):
    school: str | None = None
    degree_name: str | None = None
    field_of_study: str | None = None
    starts_at: date | None = None
    ends_at: date | None = None
    school_linkedin_profile_url: str | None = None
    logo_url: str | None = None


class LinkedInProfile(SerializableModel):
    full_name: str
    occupation: str | None
    headline: str | None
    summary: str | None
    city: str | None
    country: str | None
    public_identifier: str
    experiences: list[LinkedInExperience] = []
    education: list[LinkedInEducation] = []
    career_metrics: CareerMetrics | None = None

    def to_context_string(self) -> str:
        """Convert the profile to a formatted string context."""
        context = ""

        if self.occupation:
            context += f"Current Occupation: {self.occupation}\n\n---------\n"
        if self.headline:
            context += f"Headline: {self.headline}\n\n---------\n"
        if self.summary:
            context += f"Summary: {self.summary}\n\n---------\n"
        if self.city and self.country:
            context += f"Location of this candidate: {self.city}, {self.country}\n\n---------\n"

        for exp in self.experiences:
            context += f"Experience: {exp.title} at {exp.company}\n"
            if exp.description:
                context += f"Description: {exp.description}\n"
            if exp.starts_at:
                context += f"Start Year: {exp.starts_at.year}\n"
                context += f"Start Month: {exp.starts_at.month}\n"
            if exp.ends_at:
                context += f"End Year: {exp.ends_at.year}\n"
                context += f"End Month: {exp.ends_at.month}\n"

            if exp.company_data:
                context += exp.company_data.to_context_string()

            if exp.summarized_job_description:
                context += (
                    f"Role Summary: {exp.summarized_job_description.role_summary}\n"
                )
                context += f"Skills: {exp.summarized_job_description.skills}\n"
                context += (
                    f"Requirements: {exp.summarized_job_description.requirements}\n"
                )
            context += "\n---------\n"

        for edu in self.education:
            if edu.school and edu.degree_name and edu.field_of_study:
                context += f"Education: {edu.school}; {edu.degree_name} in {edu.field_of_study}\n"
                if edu.starts_at:
                    context += f"Start Year: {edu.starts_at.year}\n"
                    context += f"Start Month: {edu.starts_at.month}\n"
                if edu.ends_at:
                    context += f"End Year: {edu.ends_at.year}\n"
                    context += f"End Month: {edu.ends_at.month}\n"
                context += "\n---------\n"

        return context

    def dict(self, *args, **kwargs) -> dict:
        """Override dict to handle nested serialization properly."""
        # Get base dictionary
        d = super().dict(*args, **kwargs)

        # Manually serialize experiences to ensure their custom dict() method is called
        d["experiences"] = [exp.dict(*args, **kwargs) for exp in self.experiences]
        d["education"] = [edu.dict(*args, **kwargs) for edu in self.education]

        return d
