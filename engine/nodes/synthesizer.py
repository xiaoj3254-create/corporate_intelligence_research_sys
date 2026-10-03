"""synthesizer: 汇总全部 findings 起草结构化情报报告。"""
import logging

from engine.config import get_llm
from engine.prompts import SYNTHESIZER_PROMPT
from engine.state import now_str

logger = logging.getLogger(__name__)

TEAM_ORDER = {"collection": 0, "analysis": 1, "risk": 2, "deep": 3}

_TEAM_NAMES = {"collection": "情报搜集", "analysis": "综合分析",
               "risk": "风险评估", "deep": "深度研究"}


def _fallback_draft(company: str, objective: str, findings) -> str:
    """LLM 起草失败时的确定性降级稿：按团队分组罗列 findings，保证链路不中断。

    此前这里是全项目唯一没有 try/except 的 LLM 调用，无 API key / 网络故障
    会直接崩掉整图，导致前面所有 agent 的产出全部丢失。
    """
    grouped: dict[str, list] = {}
    for f in findings:
        grouped.setdefault(f.team, []).append(f)

    lines = [f"# {company} 调研简报（降级稿：LLM 起草失败，由系统自动汇总）", ""]
    if objective:
        lines += [f"**调研目标**: {objective}", ""]
    for team in ("collection", "analysis", "risk", "deep"):
        items = grouped.get(team)
        if not items:
            continue
        lines.append(f"## {_TEAM_NAMES[team]}")
        for f in items:
            src = f"（来源: {', '.join(f.source_urls[:2])}）" if f.source_urls else ""
            lines.append(f"- **[{f.agent}] {f.task}**: {f.content[:500]}{src}")
        lines.append("")
    return "\n".join(lines) if lines else f"# {company} 调研简报\n\n(无可用情报)"


def _fmt_findings(findings) -> str:
    ordered = sorted(findings, key=lambda f: TEAM_ORDER.get(f.team, 9))
    lines = []
    for f in ordered:
        src = f"（来源: {', '.join(f.source_urls[:2])}）" if f.source_urls else ""
        lines.append(f"#### [{f.team}/{f.agent}] {f.task}\n{f.content[:800]}{src}")
    return "\n\n".join(lines) or "(无)"


def synthesizer_node(state: dict) -> dict:
    company = state["company"]
    objective = state.get("objective", "")
    findings = state.get("findings") or []

    # 人工修改意见（revise 回边时存在）
    feedback = state.get("human_feedback")
    feedback_section = ""
    if feedback and feedback != "approve":
        feedback_section = f"⚠️ 人工修改意见（本稿必须落实）:\n{feedback}\n\n"

    # 检索历史报告作为结构参考（RAG 复用）
    ref_section = ""
    try:
        from knowledge.hybrid_retriever import hybrid_retrieve
        refs = hybrid_retrieve(objective or company, company=company, k=2,
                               collection_name="reports_store")
        if refs:
            ref_section = "（可参考的历史报告片段）:\n" + \
                "\n--\n".join(r["content"][:600] for r in refs) + "\n\n"
    except Exception:
        pass

    prompt = SYNTHESIZER_PROMPT.format(
        company=company, objective=objective,
        feedback_section=feedback_section, ref_section=ref_section,
        findings=_fmt_findings(findings))

    label = "重新起草" if feedback else "起草报告"
    try:
        resp = get_llm("strong").invoke(prompt)
        draft = resp.content or ""
        if not str(draft).strip():
            raise ValueError("LLM 返回空内容")
    except Exception as e:
        logger.warning("synthesizer 起草失败，降级为结构化简报: %s", e)
        draft = _fallback_draft(company, objective, findings)
        label += "（降级稿）"

    return {
        "draft_report": draft,
        "timeline": [{
            "node": "synthesizer", "agent": "analyst", "task": label,
            "status": "done", "detail": f"基于 {len(findings)} 条情报", "ts": now_str(),
        }],
    }
