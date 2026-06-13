from langchain_core.documents import Document
from app.core.vectorstore import get_vectorstore


def retrieve(query: str, top_k: int = 4) -> list[Document]:
    """Retrieve the most relevant documents for a query."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    return retriever.invoke(query)


def retrieve_with_scores(query: str, top_k: int = 4) -> list[tuple[Document, float]]:
    """Retrieve documents with similarity scores for observability."""
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search_with_score(query, k=top_k)
