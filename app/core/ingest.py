"""Ingestion pipeline: fetch (via a pluggable source) → split → embed → store.

The pipeline is domain-agnostic: it asks a `DataSource` connector for Documents,
then chunks/embeds/stores them the same way regardless of domain. Add a new
domain by adding a connector under `app/sources/` — this file doesn't change.

CLI examples:
    python -m app.core.ingest --source local
    python -m app.core.ingest --source arxiv --query "retrieval augmented generation" --max-results 30
    python -m app.core.ingest --source sec --ticker AAPL --form 10-K --max-results 3
"""

from __future__ import annotations

import argparse

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.core.vectorstore import get_vectorstore
from app.sources import available_sources, get_source


def split_documents(documents):
    """Split documents into chunks (markdown-header-aware recursive splitting)."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest(source: str = "local", **source_kwargs) -> None:
    """Full pipeline: <source>.fetch → split → embed → store."""
    print(f"=== Ingesting from source: {source} ===")
    documents = get_source(source).fetch(**source_kwargs)
    if not documents:
        print("No documents fetched.")
        return

    print(f"Fetched {len(documents)} document(s)")
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s)")

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    print("=== Ingestion complete ===")


def _cli() -> None:
    p = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base.")
    p.add_argument("--source", default="local", choices=available_sources(),
                   help="which data-source connector to use")
    # local
    p.add_argument("--docs-dir", default="docs", help="[local] directory of .md/.txt files")
    # arxiv
    p.add_argument("--query", default="retrieval augmented generation",
                   help="[arxiv] search query")
    # sec
    p.add_argument("--ticker", default="AAPL", help="[sec] company ticker")
    p.add_argument("--form", default="10-K", help="[sec] filing form type")
    # shared
    p.add_argument("--max-results", type=int, default=20,
                   help="[arxiv/sec] max documents to fetch")
    args = p.parse_args()

    if args.source == "local":
        ingest("local", docs_dir=args.docs_dir)
    elif args.source == "arxiv":
        ingest("arxiv", query=args.query, max_results=args.max_results)
    elif args.source == "sec":
        ingest("sec", ticker=args.ticker, form=args.form, max_results=args.max_results)


if __name__ == "__main__":
    _cli()
