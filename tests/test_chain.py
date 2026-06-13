"""Tests for RAG chain module."""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from app.core.chain import ask, _format_docs


@pytest.fixture
def mock_docs():
    return [
        Document(page_content="MLflow tracks experiments.", metadata={"source": "mlflow.md"}),
        Document(page_content="Models can be registered.", metadata={"source": "registry.md"}),
    ]


class TestFormatDocs:
    def test_format_docs_includes_source(self, mock_docs):
        result = _format_docs(mock_docs)
        assert "[Source: mlflow.md]" in result
        assert "[Source: registry.md]" in result

    def test_format_docs_includes_content(self, mock_docs):
        result = _format_docs(mock_docs)
        assert "MLflow tracks experiments." in result
        assert "Models can be registered." in result

    def test_format_docs_separates_with_divider(self, mock_docs):
        result = _format_docs(mock_docs)
        assert "---" in result

    def test_format_docs_handles_missing_source(self):
        docs = [Document(page_content="No source here.", metadata={})]
        result = _format_docs(docs)
        assert "[Source: unknown]" in result

    def test_format_docs_empty_list(self):
        result = _format_docs([])
        assert result == ""


class TestAsk:
    @patch("app.core.chain.build_rag_chain")
    @patch("app.core.chain.retrieve")
    def test_ask_returns_answer_and_sources(self, mock_retrieve, mock_build_chain, mock_docs):
        mock_retrieve.return_value = mock_docs
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "MLflow is used for experiment tracking."
        mock_build_chain.return_value = mock_chain

        result = ask("What is MLflow?")

        assert "answer" in result
        assert "sources" in result
        assert result["answer"] == "MLflow is used for experiment tracking."
        assert len(result["sources"]) == 2
        assert result["sources"][0]["source"] == "mlflow.md"

    @patch("app.core.chain.build_rag_chain")
    @patch("app.core.chain.retrieve")
    def test_ask_truncates_source_content(self, mock_retrieve, mock_build_chain):
        long_content = "x" * 500
        mock_retrieve.return_value = [
            Document(page_content=long_content, metadata={"source": "test.md"})
        ]
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Answer"
        mock_build_chain.return_value = mock_chain

        result = ask("test")
        assert len(result["sources"][0]["content"]) == 200
