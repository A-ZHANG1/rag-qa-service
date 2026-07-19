from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Provider: "ollama" (default, free local), "openai", or "azure"
    llm_provider: str = "ollama"

    # OpenAI
    openai_api_key: str = ""

    # Azure OpenAI (optional)
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-02-01"

    # Model settings
    llm_model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # SEC EDGAR (fair-access policy requires a descriptive User-Agent)
    sec_user_agent: str = ""

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "knowledge_base"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Agent mode: web search tool (pluggable, see docs/adr/0006)
    web_search_provider: str = "duckduckgo"  # "duckduckgo" (free, default) or "tavily"
    web_search_max_results: int = 5
    web_search_timeout_s: float = 10.0
    tavily_api_key: str = ""

    # Agent mode: supervisor routing/loop guards (see docs/adr/0005)
    agent_rag_distance_threshold: float = 0.35  # ChromaDB L2 distance; ABOVE this -> trigger web_search
    agent_max_steps: int = 4

    @property
    def use_azure(self) -> bool:
        return self.llm_provider == "azure"

    @property
    def use_ollama(self) -> bool:
        return self.llm_provider == "ollama"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
