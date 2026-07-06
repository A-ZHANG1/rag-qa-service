"""SEC EDGAR connector — fetch recent filings for a ticker via SEC's public APIs.

No API key needed, but SEC's fair-access policy REQUIRES a descriptive
User-Agent. Set `SEC_USER_AGENT` in .env (e.g., "Your Name your@email.com").

Flow: ticker → CIK (company_tickers.json) → recent filings (submissions API)
      → primary document → strip HTML → Document(s).
"""

from __future__ import annotations

from html.parser import HTMLParser

import httpx
from langchain_core.documents import Document

from app.config import get_settings
from app.sources.base import DataSource

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"


class _TextExtractor(HTMLParser):
    """Minimal stdlib HTML→text (avoids adding a bs4/lxml dependency)."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            t = data.strip()
            if t:
                self._parts.append(t)

    def text(self) -> str:
        return " ".join(self._parts)


def _strip_html(html: str) -> str:
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:  # noqa: BLE001
        pass
    return p.text()


class SECEdgarSource(DataSource):
    name = "sec"

    def _headers(self) -> dict:
        ua = get_settings().sec_user_agent or "rag-qa-service example@example.com"
        return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}

    def _resolve_cik(self, ticker: str) -> int:
        resp = httpx.get(TICKERS_URL, headers=self._headers(), timeout=30, follow_redirects=True)
        resp.raise_for_status()
        for row in resp.json().values():
            if row["ticker"].upper() == ticker.upper():
                return int(row["cik_str"])
        raise ValueError(f"ticker not found on EDGAR: {ticker}")

    def fetch(
        self,
        ticker: str = "AAPL",
        form: str = "10-K",
        max_results: int = 3,
        **kwargs,
    ) -> list[Document]:
        cik = self._resolve_cik(ticker)
        sub = httpx.get(SUBMISSIONS_URL.format(cik=cik), headers=self._headers(), timeout=30, follow_redirects=True)
        sub.raise_for_status()
        recent = sub.json()["filings"]["recent"]

        docs: list[Document] = []
        count = 0
        for i, form_type in enumerate(recent["form"]):
            if count >= max_results:
                break
            if form_type != form:
                continue
            acc_nodash = recent["accessionNumber"][i].replace("-", "")
            primary = recent["primaryDocument"][i]
            date = recent["filingDate"][i]
            url = ARCHIVE_URL.format(cik=cik, acc_nodash=acc_nodash, doc=primary)
            try:
                doc_resp = httpx.get(url, headers=self._headers(), timeout=60, follow_redirects=True)
                doc_resp.raise_for_status()
                text = _strip_html(doc_resp.text)
            except Exception as e:  # noqa: BLE001
                print(f"  skip {url}: {e}")
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": url,
                        "ticker": ticker.upper(),
                        "form": form,
                        "filing_date": date,
                        "domain": "sec",
                    },
                )
            )
            print(f"  fetched {ticker.upper()} {form} {date} ({len(text)} chars)")
            count += 1
        return docs
