"""
Three-Agent PR Workflow System — LangGraph implementation.

Inspired by a viral Maimai post: "三个agent，一个催review，一个收到comment就fix，一个应付M1"
(Three agents: one chases reviews, one auto-fixes comments, one handles the manager)

Architecture:
  Orchestrator (StateGraph) decides which agent to dispatch based on user request.
  Each agent has its own tools and prompt.

  ┌─────────────┐
  │ Orchestrator │──→ classify intent
  └──────┬──────┘
         │
    ┌────┼────┐
    ▼    ▼    ▼
  催Review  Fix   周报
  Agent   Agent  Agent
"""

from __future__ import annotations

import json
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from app.core.chain import _get_llm
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


# =====================================================
# State definition
# =====================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    agent_type: str        # "review_chaser" | "comment_fixer" | "report_writer"
    result: str            # final output from the selected agent
    tool_calls_log: list   # trace of all tool invocations


# =====================================================
# Agent 1: 催 Review Agent
# =====================================================

REVIEW_CHASER_PROMPT = """你是一个 PR Review 催促助手。你的任务是：
1. 查看所有 open 的 PR
2. 找出哪些 PR 的 review 还在 pending 或者已经等了很久
3. 给 reviewer 发送友好但有效的催促消息

注意：要礼貌但有紧迫感，提醒 reviewer PR 已经等了多少天。"""


def review_chaser_agent(state: AgentState) -> AgentState:
    """Agent that monitors PRs and reminds reviewers."""
    tool_log = []

    # Step 1: List all open PRs
    prs = list_open_prs()
    tool_log.append({"tool": "list_open_prs", "result": f"Found {len(prs)} open PRs"})

    # Step 2: Check each PR's review status
    reminders_sent = []
    for pr in prs:
        status = get_pr_review_status(pr["number"])
        tool_log.append({"tool": "get_pr_review_status", "pr": pr["number"], "state": status["review_state"]})

        # Step 3: Send reminders for pending reviews (open > 2 days)
        if status["review_state"] == "pending" and status["days_open"] >= 2:
            for reviewer in status["reviewers"]:
                result = send_review_reminder(pr["number"], reviewer)
                reminders_sent.append(result)
                tool_log.append({"tool": "send_review_reminder", "reviewer": reviewer, "pr": pr["number"]})

    # Step 4: Use LLM to generate summary
    llm = _get_llm()
    summary_prompt = f"""根据以下 PR 状态，生成一个简洁的催 review 汇总报告（中文）：

PR 列表:
{json.dumps([get_pr_review_status(pr['number']) for pr in prs], indent=2, ensure_ascii=False)}

已发送的催促:
{chr(10).join(reminders_sent) if reminders_sent else "无需催促（所有 PR 已 reviewed 或刚创建）"}

请总结：哪些 PR 需要关注，哪些已经催了，哪些不需要催。"""

    response = llm.invoke([SystemMessage(content=REVIEW_CHASER_PROMPT), HumanMessage(content=summary_prompt)])

    return {
        **state,
        "result": response.content,
        "tool_calls_log": tool_log,
        "messages": state["messages"] + [AIMessage(content=response.content)],
    }


# =====================================================
# Agent 2: Fix Comment Agent
# =====================================================

COMMENT_FIXER_PROMPT = """你是一个代码修复助手。你的任务是：
1. 查看 PR 中所有未解决的 review comments
2. 针对每个 comment 生成代码修复
3. 将所有修复作为一个 commit 推送

你要像一个资深工程师一样，理解 comment 的意图并生成高质量的修复。"""


def comment_fixer_agent(state: AgentState) -> AgentState:
    """Agent that auto-fixes PR review comments."""
    tool_log = []

    # Step 1: Find PRs with unresolved comments
    prs = list_open_prs()
    all_fixes = []
    fix_details = []

    for pr in prs:
        comments = get_unresolved_comments(pr["number"])
        tool_log.append({
            "tool": "get_unresolved_comments",
            "pr": pr["number"],
            "count": len(comments),
        })

        if not comments:
            continue

        # Step 2: Generate fix for each comment
        pr_fixes = []
        for comment in comments:
            fix = generate_code_fix(comment)
            pr_fixes.append(fix)
            tool_log.append({
                "tool": "generate_code_fix",
                "file": fix["file"],
                "line": fix["line"],
            })

        # Step 3: Push all fixes as one commit
        if pr_fixes:
            push_result = push_fix_commit(pr["number"], pr_fixes)
            tool_log.append({"tool": "push_fix_commit", "pr": pr["number"], "fixes": len(pr_fixes)})
            all_fixes.extend(pr_fixes)
            fix_details.append({
                "pr": pr["number"],
                "title": pr["title"],
                "fixes": pr_fixes,
                "push_result": push_result,
            })

    # Step 4: LLM generates summary
    llm = _get_llm()
    summary_prompt = f"""根据以下修复记录，生成一个简洁的修复汇总报告（中文）：

修复详情:
{json.dumps(fix_details, indent=2, ensure_ascii=False)}

请说明：修复了哪些 PR 的哪些问题，具体改了什么，以及修复的质量评估。"""

    response = llm.invoke([SystemMessage(content=COMMENT_FIXER_PROMPT), HumanMessage(content=summary_prompt)])

    return {
        **state,
        "result": response.content,
        "tool_calls_log": tool_log,
        "messages": state["messages"] + [AIMessage(content=response.content)],
    }


# =====================================================
# Agent 3: 应付 M1 Agent (Weekly Report)
# =====================================================

REPORT_WRITER_PROMPT = """你是一个周报生成助手。你的任务是：
1. 从 git log 中提取本周的工作内容
2. 结合 PR 状态，生成一份给 M1（经理）的周报
3. 周报要突出成果、量化影响、展示进度

风格要求：专业但不浮夸，重点突出，让经理觉得你很靠谱。"""


def report_writer_agent(state: AgentState) -> AgentState:
    """Agent that generates weekly reports for the manager."""
    tool_log = []

    # Step 1: Get recent commits
    commits = get_git_log(days=7)
    tool_log.append({"tool": "get_git_log", "commits_found": len(commits)})

    # Step 2: Get PR status
    prs = list_open_prs()
    pr_statuses = []
    for pr in prs:
        status = get_pr_review_status(pr["number"])
        pr_statuses.append(status)
        tool_log.append({"tool": "get_pr_review_status", "pr": pr["number"]})

    # Step 3: Generate structured report
    raw_report = generate_weekly_report(commits, prs)
    tool_log.append({"tool": "generate_weekly_report", "sections": 4})

    # Step 4: LLM polishes the report
    llm = _get_llm()
    polish_prompt = f"""请将以下周报润色成一封适合发给经理（M1）的邮件格式（中文），要求：
1. 开头问候
2. 本周工作亮点（用数据说话）
3. PR 进展
4. 下周计划
5. 结尾

原始周报:
{raw_report}

附加信息:
- 团队: Fabric Data Science UX
- 经理: Jeff
- 本周共 {len(commits)} 个 commits，修改了 {sum(c['files_changed'] for c in commits)} 个文件"""

    response = llm.invoke([SystemMessage(content=REPORT_WRITER_PROMPT), HumanMessage(content=polish_prompt)])

    return {
        **state,
        "result": response.content,
        "tool_calls_log": tool_log,
        "messages": state["messages"] + [AIMessage(content=response.content)],
    }


# =====================================================
# Orchestrator: route to the right agent
# =====================================================

def classify_intent(state: AgentState) -> AgentState:
    """Classify user intent and decide which agent to use."""
    user_msg = state["messages"][-1].content if state["messages"] else ""
    msg_lower = user_msg.lower()

    if any(kw in msg_lower for kw in ["review", "催", "remind", "pending", "催review"]):
        agent_type = "review_chaser"
    elif any(kw in msg_lower for kw in ["fix", "comment", "修复", "address", "resolve"]):
        agent_type = "comment_fixer"
    elif any(kw in msg_lower for kw in ["report", "周报", "m1", "manager", "status", "汇报"]):
        agent_type = "report_writer"
    else:
        # Default: use LLM to classify
        llm = _get_llm()
        classify_prompt = f"""用户说: "{user_msg}"

请判断用户意图，只回答一个单词:
- review_chaser (催促 PR review)
- comment_fixer (修复 PR comment)
- report_writer (生成周报/汇报)"""
        response = llm.invoke([HumanMessage(content=classify_prompt)])
        agent_type = response.content.strip().lower()
        if agent_type not in ("review_chaser", "comment_fixer", "report_writer"):
            agent_type = "report_writer"  # fallback

    return {**state, "agent_type": agent_type}


def route_to_agent(state: AgentState) -> Literal["review_chaser", "comment_fixer", "report_writer"]:
    """LangGraph conditional edge: route based on classified intent."""
    return state["agent_type"]


# =====================================================
# Build the LangGraph
# =====================================================

def build_workflow() -> StateGraph:
    """Build the 3-agent workflow graph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classify", classify_intent)
    workflow.add_node("review_chaser", review_chaser_agent)
    workflow.add_node("comment_fixer", comment_fixer_agent)
    workflow.add_node("report_writer", report_writer_agent)

    # Entry point
    workflow.set_entry_point("classify")

    # Conditional routing from classify → specific agent
    workflow.add_conditional_edges(
        "classify",
        route_to_agent,
        {
            "review_chaser": "review_chaser",
            "comment_fixer": "comment_fixer",
            "report_writer": "report_writer",
        },
    )

    # Each agent → END
    workflow.add_edge("review_chaser", END)
    workflow.add_edge("comment_fixer", END)
    workflow.add_edge("report_writer", END)

    return workflow.compile()


# =====================================================
# Public API
# =====================================================

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_workflow()
    return _graph


def run_agents(user_message: str) -> dict:
    """Run the 3-agent system with a user message."""
    graph = get_graph()
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_message)],
        "agent_type": "",
        "result": "",
        "tool_calls_log": [],
    }
    final_state = graph.invoke(initial_state)
    return {
        "agent_used": final_state["agent_type"],
        "result": final_state["result"],
        "tool_calls_log": final_state["tool_calls_log"],
    }
