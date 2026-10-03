"""deep_planner: 深度模式规划器。

首轮全量拆解子问题；质检未通过后的轮次只针对 coverage_gaps
生成补充子问题（避免全量重跑，控制 token 成本）。
"""
from engine.config import get_llm
from engine.prompts import DEEP_PLANNER_PROMPT, GAP_PLANNER_PROMPT
from engine.state import PlanOutput, SubQuestion, now_str


def deep_planner_node(state: dict) -> dict:
    company = state["company"]
    objective = state.get("objective", "")
    verdict = state.get("critic_verdict")
    round_no = (state.get("research_round") or 0) + 1

    if verdict and verdict.coverage_gaps:
        prompt = GAP_PLANNER_PROMPT.format(
            company=company, objective=objective,
            gaps="\n".join(f"- {g}" for g in verdict.coverage_gaps))
    else:
        prompt = DEEP_PLANNER_PROMPT.format(company=company, objective=objective)

    subs = []
    try:
        out = get_llm("strong").with_structured_output(PlanOutput).invoke(prompt)
        subs = list(out.sub_questions or [])
    except Exception:
        subs = []

    if not subs:
        subs = [SubQuestion(id=f"sq{round_no}-f", question=f"{company} 的基本情况与最新动态")]

    for i, sq in enumerate(subs[:6]):
        sq.id = sq.id or f"sq{round_no}-{i + 1}"

    label = "gap 补充规划" if (verdict and verdict.coverage_gaps) else "子问题拆解"
    return {
        "plan": subs[:6],
        "research_round": round_no,
        "timeline": [{
            "node": "deep_planner", "agent": "planner", "task": label,
            "status": "done", "detail": f"第 {round_no} 轮，{len(subs[:6])} 个子问题",
            "ts": now_str(),
        }],
    }
