"""Data-source registry — map a short name to a DataSource connector.

Add a new domain: implement a `DataSource`, register it here. Nothing else in
the pipeline changes.
"""

from __future__ import annotations

from app.sources.arxiv import ArxivSource
from app.sources.base import DataSource
from app.sources.local_docs import LocalDocsSource
from app.sources.sec_edgar import SECEdgarSource

_SOURCES: dict[str, type[DataSource]] = {
    LocalDocsSource.name: LocalDocsSource,
    ArxivSource.name: ArxivSource,
    SECEdgarSource.name: SECEdgarSource,
}


def get_source(name: str) -> DataSource:
    if name not in _SOURCES:
        raise ValueError(f"unknown source '{name}'. available: {list(_SOURCES)}")
    return _SOURCES[name]()


def available_sources() -> list[str]:
    return list(_SOURCES)


__all__ = ["DataSource", "get_source", "available_sources"]
