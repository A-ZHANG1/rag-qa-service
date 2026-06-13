"""Document ingestion pipeline: load → split → embed → store."""

import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import get_settings
from app.core.vectorstore import get_vectorstore


def _load_markdown(file_path: str) -> list[Document]:
    """Load a markdown file as a Document."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": file_path})]


LOADER_MAP = {
    ".txt": TextLoader,
    ".md": _load_markdown,
}


def load_documents(docs_dir: str = "docs"):
    """Load all supported documents from the docs directory."""
    documents = []
    for ext, loader in LOADER_MAP.items():
        pattern = os.path.join(docs_dir, f"**/*{ext}")
        for file_path in glob.glob(pattern, recursive=True):
            print(f"Loading: {file_path}")
            try:
                if callable(loader) and not isinstance(loader, type):
                    documents.extend(loader(file_path))
                else:
                    documents.extend(loader(file_path).load())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return documents


def split_documents(documents):
    """Split documents into chunks."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest(docs_dir: str = "docs"):
    """Full ingestion pipeline: load → split → embed → store."""
    print("=== Starting document ingestion ===")

    documents = load_documents(docs_dir)
    if not documents:
        print(f"No documents found in {docs_dir}/")
        return

    print(f"Loaded {len(documents)} document(s)")

    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s)")

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    print("=== Ingestion complete ===")


if __name__ == "__main__":
    ingest()
