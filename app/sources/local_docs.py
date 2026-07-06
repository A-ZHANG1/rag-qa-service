"""Local docs connector — load Markdown/txt files from a directory.

This is the original behaviour of the service, refactored behind the
`DataSource` interface so it's just one connector among many.
"""

from __future__ import annotations

import glob
import os

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from app.sources.base import DataSource


def _load_markdown(file_path: str) -> list[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": file_path, "domain": "local"})]


LOADER_MAP = {".txt": TextLoader, ".md": _load_markdown}


class LocalDocsSource(DataSource):
    name = "local"

    def fetch(self, docs_dir: str = "docs", **kwargs) -> list[Document]:
        documents: list[Document] = []
        for ext, loader in LOADER_MAP.items():
            pattern = os.path.join(docs_dir, f"**/*{ext}")
            for file_path in glob.glob(pattern, recursive=True):
                print(f"Loading: {file_path}")
                try:
                    if callable(loader) and not isinstance(loader, type):
                        documents.extend(loader(file_path))
                    else:
                        documents.extend(loader(file_path).load())
                except Exception as e:  # noqa: BLE001
                    print(f"Error loading {file_path}: {e}")
        return documents
