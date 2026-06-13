from collections.abc import AsyncIterator
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from app.config import get_settings
from app.core.retriever import retrieve

SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question based on the provided context.
If the context doesn't contain enough information to answer, say so honestly.
Always cite which part of the context you used.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def _get_llm():
    settings = get_settings()
    if settings.use_azure:
        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_deployment,
            temperature=0,
        )
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        temperature=0,
    )


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


def build_rag_chain():
    """Build the RAG chain: retrieve → format → prompt → LLM → parse."""
    llm = _get_llm()

    chain = (
        {
            "context": lambda x: _format_docs(retrieve(x["question"])),
            "question": RunnablePassthrough() | (lambda x: x["question"]),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str) -> dict:
    """Synchronous RAG query. Returns answer and source documents."""
    docs = retrieve(question)
    chain = build_rag_chain()
    answer = chain.invoke({"question": question})
    return {
        "answer": answer,
        "sources": [
            {
                "content": doc.page_content[:200],
                "source": doc.metadata.get("source", "unknown"),
            }
            for doc in docs
        ],
    }


async def ask_stream(question: str) -> AsyncIterator[str]:
    """Streaming RAG query. Yields answer chunks."""
    chain = build_rag_chain()
    async for chunk in chain.astream({"question": question}):
        yield chunk
