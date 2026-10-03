"""intake: 任务解析与模式分流入口。"""
from engine.config import get_llm
from engine.prompts import INTAKE_PROMPT
from engine.state import now_str


def intake_node(state: dict) -> dict:
    company = state["company"]
    objective = (state.get("objective") or "").strip()
    mode = state.get("mode") or "standard"

    # 用轻模型提炼调研目标（失败则原样透传，不阻塞主链路）
    refined = objective
    if objective:
        try:
            resp = get_llm("cheap").invoke(
                INTAKE_PROMPT.format(company=company, objective=objective))
            text = (resp.content or "").strip()
            if text:
                refined = text[:300]
        except Exception:
            pass
    if not refined:
        refined = f"对 {company} 进行全面的企业情报调研"

    return {
        "objective": refined,
        "mode": mode,
        "research_round": 0,
        "timeline": [{
            "node": "intake", "agent": "router", "task": "任务解析",
            "status": "done", "detail": f"模式: {mode}", "ts": now_str(),
        }],
    }
