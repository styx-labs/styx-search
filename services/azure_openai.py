from typing import Optional
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from agent.get_secret import get_secret


llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    openai_api_version="2024-08-01-preview",
    azure_endpoint=get_secret("azure-openai-endpoint", "1"),
    openai_api_key=get_secret("azure-openai-api-key", "1"),
    temperature=0,
)

llm_4o_mini = AzureChatOpenAI(
    deployment_name="gpt-4o-mini",
    openai_api_version="2024-08-01-preview",
    azure_endpoint=get_secret("azure-openai-endpoint", "1"),
    openai_api_key=get_secret("azure-openai-api-key", "1"),
    temperature=0,
)


def get_azure_openai() -> Optional[AzureOpenAI]:
    return AzureOpenAI(
        api_key=get_secret("azure-openai-api-key", "1"),
        api_version="2024-08-01-preview",
        azure_endpoint=get_secret("azure-openai-endpoint", "1"),
    )
