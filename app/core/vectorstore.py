import chromadb
from langchain_chroma import Chroma
from app.config import get_settings
from app.core.embeddings import get_embedding_model


_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    """Get or create the ChromaDB vector store singleton."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    embedding_model = get_embedding_model()

    _vectorstore = Chroma(
        client=client,
        collection_name=settings.collection_name,
        embedding_function=embedding_model,
    )
    return _vectorstore
