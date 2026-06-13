from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from app.config import get_settings


def get_embedding_model():
    """Create embedding model based on configuration."""
    settings = get_settings()

    if settings.use_azure:
        return AzureOpenAIEmbeddings(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            model=settings.embedding_model,
        )

    return OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )
