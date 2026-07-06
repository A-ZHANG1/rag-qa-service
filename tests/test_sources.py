import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.sources import available_sources, get_source  # noqa: E402
from app.sources.base import DataSource  # noqa: E402
from app.sources.local_docs import LocalDocsSource  # noqa: E402


def test_registry_lists_all_sources():
    assert set(available_sources()) == {"local", "arxiv", "sec"}


def test_get_source_returns_datasource_instances():
    for name in available_sources():
        src = get_source(name)
        assert isinstance(src, DataSource)
        assert src.name == name


def test_unknown_source_raises():
    try:
        get_source("does-not-exist")
        raise AssertionError("expected ValueError for unknown source")
    except ValueError:
        pass


def test_local_source_reads_markdown(tmp_path):
    (tmp_path / "a.md").write_text("# Hello\n\nworld", encoding="utf-8")
    docs = LocalDocsSource().fetch(docs_dir=str(tmp_path))
    assert len(docs) == 1
    assert "Hello" in docs[0].page_content
    assert docs[0].metadata["domain"] == "local"
