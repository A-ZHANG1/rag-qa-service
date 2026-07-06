"""arXiv connector — fetch paper titles + abstracts via the public arXiv API.

No API key needed. API: http://export.arxiv.org/api/query (returns Atom XML).
Great for an AI-focused knowledge base ("chat with the latest papers on X").
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx
from langchain_core.documents import Document

from app.sources.base import DataSource

ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"


class ArxivSource(DataSource):
    name = "arxiv"

    def fetch(
        self,
        query: str = "retrieval augmented generation",
        max_results: int = 20,
        **kwargs,
    ) -> list[Document]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = httpx.get(
            ARXIV_API,
            params=params,
            timeout=30,
            follow_redirects=True,
            headers={"User-Agent": "rag-qa-service/0.1 (+https://github.com/A-ZHANG1)"},
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        docs: list[Document] = []
        for entry in root.findall(f"{_ATOM}entry"):
            title = (entry.findtext(f"{_ATOM}title") or "").strip().replace("\n", " ")
            summary = (entry.findtext(f"{_ATOM}summary") or "").strip()
            url = (entry.findtext(f"{_ATOM}id") or "").strip()
            published = (entry.findtext(f"{_ATOM}published") or "").strip()
            authors = ", ".join(
                (a.findtext(f"{_ATOM}name") or "").strip()
                for a in entry.findall(f"{_ATOM}author")
            )
            content = (
                f"Title: {title}\n"
                f"Authors: {authors}\n"
                f"Published: {published}\n\n"
                f"Abstract:\n{summary}"
            )
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": url,
                        "title": title,
                        "published": published,
                        "domain": "arxiv",
                    },
                )
            )
            print(f"  fetched: {title[:70]}")
        return docs
