"""
GitHub PR Workflow Tools — the "hands and feet" of the 3-agent system.

Simulates GitHub API interactions. In production, replace with real
PyGithub / `gh` CLI calls.
"""

from datetime import datetime, timedelta
import random


# ---- Simulated GitHub state ----

MOCK_PRS = [
    {
        "number": 142,
        "title": "feat: add GenAI tracing UI",
        "author": "wanyuezhang",
        "reviewers": ["jeff-zheng", "alice-msft"],
        "status": "open",
        "review_state": "pending",
        "created_at": (datetime.now() - timedelta(days=3)).isoformat(),
        "comments": [],
    },
    {
        "number": 138,
        "title": "fix: MLflow autolog metric rendering",
        "author": "wanyuezhang",
        "reviewers": ["bob-msft"],
        "status": "open",
        "review_state": "changes_requested",
        "created_at": (datetime.now() - timedelta(days=5)).isoformat(),
        "comments": [
            {
                "id": 1,
                "author": "bob-msft",
                "body": "Please add null check for metricValue on line 45.",
                "file": "src/components/MetricChart.tsx",
                "line": 45,
                "resolved": False,
            },
            {
                "id": 2,
                "author": "bob-msft",
                "body": "This useEffect dependency array is missing `traceId`.",
                "file": "src/hooks/useTraceData.ts",
                "line": 23,
                "resolved": False,
            },
        ],
    },
    {
        "number": 135,
        "title": "chore: upgrade E2E test framework",
        "author": "wanyuezhang",
        "reviewers": ["jeff-zheng"],
        "status": "open",
        "review_state": "approved",
        "created_at": (datetime.now() - timedelta(days=7)).isoformat(),
        "comments": [],
    },
]


def list_open_prs(author: str = "wanyuezhang") -> list[dict]:
    """List all open PRs by author."""
    return [pr for pr in MOCK_PRS if pr["author"] == author and pr["status"] == "open"]


def get_pr_review_status(pr_number: int) -> dict:
    """Get review status for a specific PR."""
    for pr in MOCK_PRS:
        if pr["number"] == pr_number:
            return {
                "number": pr["number"],
                "title": pr["title"],
                "review_state": pr["review_state"],
                "reviewers": pr["reviewers"],
                "pending_comments": [c for c in pr["comments"] if not c["resolved"]],
                "days_open": (datetime.now() - datetime.fromisoformat(pr["created_at"])).days,
            }
    return {"error": f"PR #{pr_number} not found"}


def send_review_reminder(pr_number: int, reviewer: str) -> str:
    """Send a review reminder to a reviewer (simulated)."""
    for pr in MOCK_PRS:
        if pr["number"] == pr_number:
            return (
                f"✅ Sent reminder to @{reviewer} for PR #{pr_number} "
                f"(\"{pr['title']}\"):\n"
                f"\"Hi @{reviewer}, friendly reminder — this PR has been "
                f"waiting for review for {(datetime.now() - datetime.fromisoformat(pr['created_at'])).days} days. "
                f"Could you take a look when you have a moment? Thanks! 🙏\""
            )
    return f"❌ PR #{pr_number} not found"


def get_unresolved_comments(pr_number: int) -> list[dict]:
    """Get all unresolved review comments for a PR."""
    for pr in MOCK_PRS:
        if pr["number"] == pr_number:
            return [c for c in pr["comments"] if not c["resolved"]]
    return []


def generate_code_fix(comment: dict) -> dict:
    """Generate a code fix for a review comment (simulated LLM fix)."""
    fixes = {
        "Please add null check for metricValue on line 45.": {
            "file": comment.get("file", "unknown"),
            "line": comment.get("line", 0),
            "original": "const value = metricValue.toFixed(2);",
            "fixed": "const value = metricValue != null ? metricValue.toFixed(2) : 'N/A';",
            "explanation": "Added null check with fallback to 'N/A'",
        },
        "This useEffect dependency array is missing `traceId`.": {
            "file": comment.get("file", "unknown"),
            "line": comment.get("line", 0),
            "original": "useEffect(() => { fetchTrace(); }, []);",
            "fixed": "useEffect(() => { fetchTrace(); }, [traceId]);",
            "explanation": "Added `traceId` to dependency array to trigger re-fetch",
        },
    }
    body = comment.get("body", "")
    if body in fixes:
        return fixes[body]
    return {
        "file": comment.get("file", "unknown"),
        "line": comment.get("line", 0),
        "original": "// original code",
        "fixed": "// AI-generated fix based on comment",
        "explanation": f"Auto-fix for: {body}",
    }


def push_fix_commit(pr_number: int, fixes: list[dict]) -> str:
    """Push code fixes as a new commit (simulated)."""
    file_list = ", ".join(set(f["file"] for f in fixes))
    return (
        f"✅ Pushed commit to PR #{pr_number}:\n"
        f"  commit abc{random.randint(1000,9999)}\n"
        f"  Author: AI-Fix-Bot\n"
        f"  Message: \"fix: address review comments\"\n"
        f"  Files changed: {file_list}\n"
        f"  {len(fixes)} fix(es) applied"
    )


def get_git_log(author: str = "wanyuezhang", days: int = 7) -> list[dict]:
    """Get recent git commits by author (simulated)."""
    return [
        {
            "hash": "a1b2c3d",
            "message": "feat: add GenAI tracing span detail panel",
            "date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "files_changed": 4,
        },
        {
            "hash": "e4f5g6h",
            "message": "fix: MLflow metric chart null pointer",
            "date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
            "files_changed": 2,
        },
        {
            "hash": "i7j8k9l",
            "message": "test: add E2E tests for trace list pagination",
            "date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
            "files_changed": 3,
        },
        {
            "hash": "m0n1o2p",
            "message": "feat: implement AOAI tenant settings toggle UI",
            "date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
            "files_changed": 6,
        },
        {
            "hash": "q3r4s5t",
            "message": "refactor: extract shared trace formatter utility",
            "date": (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d"),
            "files_changed": 2,
        },
    ]


def generate_weekly_report(commits: list[dict], prs: list[dict]) -> str:
    """Generate a weekly status report."""
    report_lines = [
        f"📊 **Weekly Report** ({datetime.now().strftime('%Y-%m-%d')})",
        "",
        "## Commits This Week",
    ]
    for c in commits:
        report_lines.append(f"- `{c['hash']}` {c['message']} ({c['files_changed']} files)")

    report_lines.append("")
    report_lines.append("## PR Status")
    for pr in prs:
        status_emoji = {"pending": "⏳", "changes_requested": "🔄", "approved": "✅"}.get(
            pr["review_state"], "❓"
        )
        report_lines.append(f"- {status_emoji} PR #{pr['number']}: {pr['title']}")

    report_lines.extend([
        "",
        "## Key Accomplishments",
        "- Completed GenAI tracing span detail panel (feature)",
        "- Fixed MLflow metric chart null pointer (bug fix)",
        "- Added E2E tests for trace list pagination (quality)",
        "",
        "## Next Week Plan",
        "- Continue DataAgent evaluation UI integration",
        "- Address remaining PR review comments",
    ])
    return "\n".join(report_lines)
