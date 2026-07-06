"""Pluggable data-source connectors.

Every domain (local docs, SEC filings, arXiv papers, ...) implements the SAME
`DataSource` interface: given source-specific params, return a list of LangChain
`Document`s. Everything downstream — chunk → embed → store → retrieve → generate
→ eval — is domain-agnostic and reused across all sources.

This is the "component reuse" seam: **add a new domain = add one connector,
touch nothing else.** See docs/adr/0004-pluggable-data-source-connectors.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class DataSource(ABC):
    """A pluggable knowledge source that yields Documents for ingestion."""

    #: short id used on the CLI / in the registry, e.g. "arxiv"
    name: str = "base"

    @abstractmethod
    def fetch(self, **kwargs) -> list[Document]:
        """Fetch documents for this source. `kwargs` are source-specific.

        Each returned Document should carry useful metadata (at minimum a
        ``source`` URL/path and a ``domain`` tag) so retrieval results can cite
        provenance.
        """
        raise NotImplementedError
