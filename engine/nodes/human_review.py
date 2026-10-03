"""human_review: human-in-the-loop 人工审批节点（interrupt 暂停，等 Command(resume) 恢复）。"""
from langgraph.types import interrupt

from engine.state import now_str


def human_review_node(state: dict) -> dict:
    # 评估/演示场景可跳过审批
    if state.get("auto_approve"):
        return {
            "human_feedback": "approve",
            "timeline": [{
                "node": "human_review", "agent": "审批", "task": "人工审批",
                "status": "done", "detail": "自动通过（auto_approve）", "ts": now_str(),
            }],
        }

    findings = state.get("findings") or []
    risk_flags = [f.content[:200] for f in findings
                  if f.team == "risk" and f.credibility != "low"]

    # 暂停图执行，等待人工决策；resume 值即 human_feedback
    decision = interrupt({
        "draft": state.get("draft_report") or "",
        "risk_flags": risk_flags,
    })

    return {
        "human_feedback": decision,
        "timeline": [{
            "node": "human_review", "agent": "审批", "task": "人工审批",
            "status": "done", "detail": f"决策: {str(decision)[:80]}", "ts": now_str(),
        }],
    }


def human_gate(state: dict) -> str:
    return "approve" if state.get("human_feedback") == "approve" else "revise"
