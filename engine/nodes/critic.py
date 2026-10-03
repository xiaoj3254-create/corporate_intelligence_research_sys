"""critic: 深度模式反思质检（覆盖度/证据/矛盾/时效四维）。"""
from engine.config import get_llm, settings
from engine.prompts import CRITIC_PROMPT
from engine.state import QualityAssessment, now_str


def _fmt_findings(findings) -> str:
    lines = []
    for f in findings:
        src = f"（来源: {', '.join(f.source_urls[:2])}）" if f.source_urls else ""
        lines.append(f"[{f.agent}]{f.task}\n{f.content[:500]}{src}")
    return "\n\n".join(lines) or "(无)"


def critic_node(state: dict) -> dict:
    plan = state.get("plan") or []
    findings = state.get("findings") or []
    prompt = CRITIC_PROMPT.format(
        objective=state.get("objective", ""),
        sub_questions="\n".join(f"- [{q.id}] {q.question}" for q in plan) or "(无)",
        findings=_fmt_findings(findings))

    try:
        verdict = get_llm("strong").with_structured_output(QualityAssessment).invoke(prompt)
    except Exception as e:
        # 质检异常时放行，避免死循环
        verdict = QualityAssessment(overall_pass=True,
                                    contradictions=[f"质检异常(放行): {e}"])

    detail = "通过" if verdict.overall_pass else \
        f"不通过，缺口 {len(verdict.coverage_gaps)} 项"
    return {
        "critic_verdict": verdict,
        "timeline": [{
            "node": "critic", "agent": "critic", "task": "反思质检",
            "status": "done", "detail": detail, "ts": now_str(),
        }],
    }


def critic_gate(state: dict) -> str:
    """反思门控: 通过 / 达到轮次上限 / 无具体缺口 → pass；否则 revise 回 deep_planner。"""
    v = state.get("critic_verdict")
    if not v or v.overall_pass:
        return "pass"
    if (state.get("research_round") or 0) >= settings.max_research_rounds:
        return "pass"
    if not v.coverage_gaps:
        return "pass"
    return "revise"
