"""Tests for the 3-agent PR workflow system."""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage

from app.agents.tools import (
    list_open_prs,
    get_pr_review_status,
    send_review_reminder,
    get_unresolved_comments,
    generate_code_fix,
    push_fix_commit,
    get_git_log,
    generate_weekly_report,
)
from app.agents.workflow import (
    classify_intent,
    review_chaser_agent,
    comment_fixer_agent,
    report_writer_agent,
    run_agents,
    AgentState,
)
from langchain_core.messages import HumanMessage


# =====================================================
# Tool tests
# =====================================================

class TestTools:
    def test_list_open_prs(self):
        prs = list_open_prs()
        assert len(prs) == 3
        assert all(pr["status"] == "open" for pr in prs)
        assert all(pr["author"] == "wanyuezhang" for pr in prs)

    def test_get_pr_review_status(self):
        status = get_pr_review_status(142)
        assert status["number"] == 142
        assert status["review_state"] == "pending"
        assert "jeff-zheng" in status["reviewers"]

    def test_get_pr_review_status_not_found(self):
        status = get_pr_review_status(9999)
        assert "error" in status

    def test_send_review_reminder(self):
        result = send_review_reminder(142, "jeff-zheng")
        assert "jeff-zheng" in result
        assert "PR #142" in result
        assert "✅" in result

    def test_get_unresolved_comments(self):
        comments = get_unresolved_comments(138)
        assert len(comments) == 2
        assert comments[0]["author"] == "bob-msft"

    def test_get_unresolved_comments_none(self):
        comments = get_unresolved_comments(142)
        assert len(comments) == 0

    def test_generate_code_fix_known(self):
        comment = {
            "body": "Please add null check for metricValue on line 45.",
            "file": "src/components/MetricChart.tsx",
            "line": 45,
        }
        fix = generate_code_fix(comment)
        assert fix["file"] == "src/components/MetricChart.tsx"
        assert "null" in fix["fixed"].lower() or "!= null" in fix["fixed"]

    def test_push_fix_commit(self):
        fixes = [{"file": "test.ts", "line": 1, "original": "x", "fixed": "y"}]
        result = push_fix_commit(138, fixes)
        assert "PR #138" in result
        assert "1 fix" in result

    def test_get_git_log(self):
        commits = get_git_log(days=7)
        assert len(commits) == 5
        assert all("hash" in c and "message" in c for c in commits)

    def test_generate_weekly_report(self):
        commits = get_git_log()
        prs = list_open_prs()
        report = generate_weekly_report(commits, prs)
        assert "Weekly Report" in report
        assert "PR Status" in report


# =====================================================
# Intent classification tests
# =====================================================

class TestClassifyIntent:
    def _make_state(self, msg: str) -> AgentState:
        return {
            "messages": [HumanMessage(content=msg)],
            "agent_type": "",
            "result": "",
            "tool_calls_log": [],
        }

    def test_classify_review(self):
        state = classify_intent(self._make_state("帮我催一下 review"))
        assert state["agent_type"] == "review_chaser"

    def test_classify_fix(self):
        state = classify_intent(self._make_state("fix all PR comments"))
        assert state["agent_type"] == "comment_fixer"

    def test_classify_report(self):
        state = classify_intent(self._make_state("生成本周周报给 M1"))
        assert state["agent_type"] == "report_writer"

    def test_classify_review_en(self):
        state = classify_intent(self._make_state("remind reviewers"))
        assert state["agent_type"] == "review_chaser"


# =====================================================
# Agent tests (with mocked LLM)
# =====================================================

def _mock_llm():
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content="Mock agent response")
    return mock


class TestAgents:
    def _make_state(self, msg: str, agent_type: str) -> AgentState:
        return {
            "messages": [HumanMessage(content=msg)],
            "agent_type": agent_type,
            "result": "",
            "tool_calls_log": [],
        }

    @patch("app.agents.workflow._get_llm", return_value=_mock_llm())
    def test_review_chaser(self, mock_llm):
        state = self._make_state("催 review", "review_chaser")
        result = review_chaser_agent(state)
        assert result["result"] == "Mock agent response"
        assert len(result["tool_calls_log"]) > 0
        tools_used = {log["tool"] for log in result["tool_calls_log"]}
        assert "list_open_prs" in tools_used
        assert "get_pr_review_status" in tools_used

    @patch("app.agents.workflow._get_llm", return_value=_mock_llm())
    def test_comment_fixer(self, mock_llm):
        state = self._make_state("fix comments", "comment_fixer")
        result = comment_fixer_agent(state)
        assert result["result"] == "Mock agent response"
        tools_used = {log["tool"] for log in result["tool_calls_log"]}
        assert "get_unresolved_comments" in tools_used
        assert "generate_code_fix" in tools_used
        assert "push_fix_commit" in tools_used

    @patch("app.agents.workflow._get_llm", return_value=_mock_llm())
    def test_report_writer(self, mock_llm):
        state = self._make_state("周报", "report_writer")
        result = report_writer_agent(state)
        assert result["result"] == "Mock agent response"
        tools_used = {log["tool"] for log in result["tool_calls_log"]}
        assert "get_git_log" in tools_used
        assert "generate_weekly_report" in tools_used


# =====================================================
# Integration: run_agents (mocked LLM)
# =====================================================

class TestRunAgents:
    @patch("app.agents.workflow._get_llm", return_value=_mock_llm())
    def test_run_review_chaser(self, mock_llm):
        result = run_agents("帮我催一下 review")
        assert result["agent_used"] == "review_chaser"
        assert result["result"]
        assert len(result["tool_calls_log"]) > 0

    @patch("app.agents.workflow._get_llm", return_value=_mock_llm())
    def test_run_comment_fixer(self, mock_llm):
        result = run_agents("把 comment 都修了")
        assert result["agent_used"] == "comment_fixer"

    @patch("app.agents.workflow._get_llm", return_value=_mock_llm())
    def test_run_report_writer(self, mock_llm):
        result = run_agents("生成周报给 manager")
        assert result["agent_used"] == "report_writer"
