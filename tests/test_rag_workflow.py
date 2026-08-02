"""Tests for the RAG Agent mode workflow (supervisor + specialist).

Covers the 3 key paths from docs/adr/0005-rag-agent-supervisor-pattern.md:
local docs sufficient (no web trigger), local docs insufficient (web
triggers and succeeds), and web_specialist failing (must still return a
valid degraded answer instead of raising/hanging) -- the last one directly
validates the "multi-agent collaboration must not fail to produce a
result" property this design exists for.
"""

from unittest.mock import MagicMock, patch

from app.agents import rag_workflow


class FakeLLMResponse:
    """Minimal stand-in for a LangChain chat model response (has .content)."""

    def __init__(self, content: str):
        self.content = content


def make_fake_llm(reply_text: str = "a synthesized answer") -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = FakeLLMResponse(reply_text)
    return llm


class TestLocalDocsSufficient:
    @patch("app.agents.rag_workflow.web_search")
    @patch("app.agents.rag_workflow.rag_search")
    @patch("app.agents.rag_workflow._get_llm")
    def test_web_specialist_is_skipped(self, mock_get_llm, mock_rag_search, mock_web_search):
        mock_get_llm.return_value = make_fake_llm("answer from local docs")
        mock_rag_search.return_value = {
            "success": True,
            "results": [{"content": "MLflow is an MLOps platform.", "source": "docs/mlflow.md", "score": 0.1}],
            "min_distance": 0.1,
        }

        result = rag_workflow.run_agent("What is MLflow?")

        assert mock_web_search.called is False
        assert result["degraded"] is False
        assert result["answer"] == "answer from local docs"
        assert [s["step"] for s in result["trace"]] == ["rag_specialist", "synthesis"]
        assert result["sources"][0]["origin"] == "rag"

    @patch("app.agents.rag_workflow.web_search")
    @patch("app.agents.rag_workflow.rag_search")
    @patch("app.agents.rag_workflow._get_llm")
    def test_low_distance_below_threshold_does_not_trigger_web(
        self, mock_get_llm, mock_rag_search, mock_web_search
    ):
        mock_get_llm.return_value = make_fake_llm()
        mock_rag_search.return_value = {
            "success": True,
            "results": [{"content": "x", "source": "a.md", "score": 0.34}],
            "min_distance": 0.34,  # just under the default 0.35 threshold
        }

        rag_workflow.run_agent("test")

        assert mock_web_search.called is False


class TestLocalDocsInsufficient:
    @patch("app.agents.rag_workflow.web_search")
    @patch("app.agents.rag_workflow.rag_search")
    @patch("app.agents.rag_workflow._get_llm")
    def test_web_specialist_triggers_and_succeeds(self, mock_get_llm, mock_rag_search, mock_web_search):
        mock_get_llm.return_value = make_fake_llm("answer combining both sources")
        mock_rag_search.return_value = {"success": True, "results": [], "min_distance": 999.0}
        mock_web_search.return_value = {
            "success": True,
            "results": [
                {"title": "MLflow 3.0 changelog", "snippet": "New features...", "url": "https://mlflow.org/changelog"}
            ],
            "provider": "duckduckgo",
        }

        result = rag_workflow.run_agent("What's new in MLflow 3.0?")

        assert mock_web_search.called is True
        assert result["degraded"] is False
        assert [s["step"] for s in result["trace"]] == ["rag_specialist", "web_specialist", "synthesis"]
        assert result["sources"][0]["origin"] == "web"

    @patch("app.agents.rag_workflow.web_search")
    @patch("app.agents.rag_workflow.rag_search")
    @patch("app.agents.rag_workflow._get_llm")
    def test_empty_rag_index_triggers_web(self, mock_get_llm, mock_rag_search, mock_web_search):
        """An uninitialized/empty index (min_distance=inf) must also trigger web search."""
        mock_get_llm.return_value = make_fake_llm()
        mock_rag_search.return_value = {"success": True, "results": [], "min_distance": float("inf")}
        mock_web_search.return_value = {"success": True, "results": [], "provider": "duckduckgo"}

        rag_workflow.run_agent("test")

        assert mock_web_search.called is True


class TestWebSpecialistFailure:
    """The critical reliability path: a failed/timed-out web search must
    still produce a real answer instead of the request raising or hanging."""

    @patch("app.agents.rag_workflow.web_search")
    @patch("app.agents.rag_workflow.rag_search")
    @patch("app.agents.rag_workflow._get_llm")
    def test_web_failure_degrades_gracefully(self, mock_get_llm, mock_rag_search, mock_web_search):
        mock_get_llm.return_value = make_fake_llm("degraded answer, local docs only")
        mock_rag_search.return_value = {"success": True, "results": [], "min_distance": 999.0}
        mock_web_search.return_value = {
            "success": False,
            "error": "timed out after 10.0s",
            "results": [],
        }

        result = rag_workflow.run_agent("What's new in MLflow 3.0?")

        assert mock_web_search.called is True
        assert result["degraded"] is True, "degraded flag must be set when web_search fails"
        assert result["answer"] == "degraded answer, local docs only", "must still produce a real answer"
        assert [s["step"] for s in result["trace"]] == ["rag_specialist", "web_specialist", "synthesis"]

    @patch("app.agents.rag_workflow.web_search")
    @patch("app.agents.rag_workflow.rag_search")
    @patch("app.agents.rag_workflow._get_llm")
    def test_both_specialists_empty_still_returns_answer(self, mock_get_llm, mock_rag_search, mock_web_search):
        """Worst case: local docs empty AND web search fails -- synthesis must
        still run and return a real (if unhelpful) answer, never raise."""
        mock_get_llm.return_value = make_fake_llm("I don't have enough information to answer that.")
        mock_rag_search.return_value = {"success": False, "error": "index not found", "results": [], "min_distance": float("inf")}
        mock_web_search.return_value = {"success": False, "error": "network unreachable", "results": []}

        result = rag_workflow.run_agent("test")

        assert result["degraded"] is True
        assert result["answer"]
        assert result["sources"] == []


class TestStepCap:
    @patch("app.agents.rag_workflow.web_search")
    @patch("app.agents.rag_workflow.rag_search")
    @patch("app.agents.rag_workflow._get_llm")
    @patch("app.agents.rag_workflow.get_settings")
    def test_step_cap_skips_web_even_when_rag_insufficient(
        self, mock_get_settings, mock_get_llm, mock_rag_search, mock_web_search
    ):
        """agent_max_steps=1 means the cap is already hit after rag_specialist's
        first step, so web_specialist must be skipped regardless of confidence."""
        settings = MagicMock()
        settings.agent_max_steps = 1
        settings.agent_rag_distance_threshold = 0.35
        mock_get_settings.return_value = settings
        mock_get_llm.return_value = make_fake_llm()
        mock_rag_search.return_value = {"success": True, "results": [], "min_distance": 999.0}

        rag_workflow.run_agent("test")

        assert mock_web_search.called is False
