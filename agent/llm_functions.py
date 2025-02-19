from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from services.llms import llm_fast
from models.base import RolesOutput
from agent.prompts import identify_roles_prompt


@traceable(name="identify_roles")
def identify_roles(candidate_profile: str) -> RolesOutput:
    """Extract roles from candidate profile."""
    structured_llm = llm_fast.with_structured_output(RolesOutput)
    output = structured_llm.invoke(
        [
            SystemMessage(content=identify_roles_prompt),
            HumanMessage(content=candidate_profile),
        ]
    )
    return output
