"""Tests for API endpoints."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestRootEndpoint:
    def test_root_returns_message(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "RAG QA Service" in response.json()["message"]


class TestChatEndpoint:
    @patch("app.api.routes.ask")
    def test_chat_returns_answer(self, mock_ask):
        mock_ask.return_value = {
            "answer": "MLflow is a platform for ML lifecycle.",
            "sources": [{"content": "MLflow is...", "source": "mlflow.md"}],
        }

        response = client.post("/api/v1/chat", json={"query": "What is MLflow?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert data["answer"] == "MLflow is a platform for ML lifecycle."

    def test_chat_rejects_empty_query(self):
        response = client.post("/api/v1/chat", json={"query": ""})
        assert response.status_code == 422

    def test_chat_rejects_missing_query(self):
        response = client.post("/api/v1/chat", json={})
        assert response.status_code == 422

    @patch("app.api.routes.ask")
    def test_chat_with_custom_top_k(self, mock_ask):
        mock_ask.return_value = {"answer": "test", "sources": []}
        response = client.post("/api/v1/chat", json={"query": "test", "top_k": 2})
        assert response.status_code == 200
