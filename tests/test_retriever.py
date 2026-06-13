"""Tests for retriever module."""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from app.core.retriever import retrieve, retrieve_with_scores


@pytest.fixture
def mock_docs():
    return [
        Document(page_content="MLflow is an open-source platform.", metadata={"source": "mlflow.md"}),
        Document(page_content="RAG enhances LLMs with retrieval.", metadata={"source": "rag.md"}),
    ]


@pytest.fixture
def mock_vectorstore(mock_docs):
    vs = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = mock_docs
    vs.as_retriever.return_value = retriever
    vs.similarity_search_with_score.return_value = [
        (mock_docs[0], 0.85),
        (mock_docs[1], 0.72),
    ]
    return vs


class TestRetrieve:
    @patch("app.core.retriever.get_vectorstore")
    def test_retrieve_returns_documents(self, mock_get_vs, mock_vectorstore, mock_docs):
        mock_get_vs.return_value = mock_vectorstore
        results = retrieve("What is MLflow?")

        assert len(results) == 2
        assert results[0].page_content == "MLflow is an open-source platform."
        assert results[0].metadata["source"] == "mlflow.md"

    @patch("app.core.retriever.get_vectorstore")
    def test_retrieve_respects_top_k(self, mock_get_vs, mock_vectorstore):
        mock_get_vs.return_value = mock_vectorstore
        retrieve("test query", top_k=2)

        mock_vectorstore.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 2},
        )

    @patch("app.core.retriever.get_vectorstore")
    def test_retrieve_with_scores_returns_tuples(self, mock_get_vs, mock_vectorstore, mock_docs):
        mock_get_vs.return_value = mock_vectorstore
        results = retrieve_with_scores("What is RAG?")

        assert len(results) == 2
        doc, score = results[0]
        assert doc.page_content == "MLflow is an open-source platform."
        assert score == 0.85
