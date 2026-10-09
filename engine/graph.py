"""主图组装：B(分层团队) + C(深度研究) 双模式融合。

调度语义:
- standard: 搜集组 → Send 固定 fan-out ×2（分析组 ∥ 风控组，DAG 依赖: 后者消费前者产出）
- deep:     deep_planner → Send 动态 fan-out ×N researcher → critic 反思门控
            （revise 只针对 coverage_gaps 补充子问题，最多 max_research_rounds 轮）
- 汇合:     synthesizer → human_review(interrupt) → reporter → END
"""
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from engine.agents.researcher import researcher_node
from engine.nodes.critic import critic_gate, critic_node
from engine.nodes.deep_planner import deep_planner_node
from engine.nodes.human_review import human_gate, human_review_node
from engine.nodes.intake import intake_node
from engine.nodes.reporter import reporter_node
from engine.nodes.synthesizer import synthesizer_node
from engine.state import MainState
from engine.teams.analysis_team import analysis_team_node
from engine.teams.collection_team import collection_team_node
from engine.teams.risk_team import risk_team_node


def _fan_out_analysis_risk(state: dict):
    """B 模式固定 fan-out：分析组与风控组并行，各自携带搜集组的 findings 作为上游输入。

    task_id 必须随 Send 下发：wrap_team 的事件实时外推（sink）依赖它定位任务。
    """
    base = {
        "company": state["company"],
        "objective": state.get("objective", ""),
        "prior_findings": [f for f in state.get("findings") or []
                           if f.team == "collection"],
        "task_id": state.get("task_id", ""),
    }
    return [Send("analysis_team", {**base}), Send("risk_team", {**base})]


def _fan_out_researchers(state: dict):
    """C 模式动态 fan-out：子问题数量运行时决定，开 N 个并行 researcher。

    task_id 必须随 Send 下发：researcher 的事件实时外推（sink）依赖它定位任务。
    """
    return [Send("researcher", {
        "company": state["company"],
        "objective": state.get("objective", ""),
        "sub_q": sq,
        "task_id": state.get("task_id", ""),
    }) for sq in state.get("plan") or []]


def build_graph(checkpointer=None):
    builder = StateGraph(MainState)

    builder.add_node("intake", intake_node)
    builder.add_node("collection_team", collection_team_node)
    builder.add_node("analysis_team", analysis_team_node)
    builder.add_node("risk_team", risk_team_node)
    builder.add_node("deep_planner", deep_planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("critic", critic_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("reporter", reporter_node)

    builder.add_edge(START, "intake")

    # 模式分流
    builder.add_conditional_edges(
        "intake",
        lambda s: "deep" if s.get("mode") == "deep" else "standard",
        {"standard": "collection_team", "deep": "deep_planner"})

    # B 模式 DAG：搜集 → (分析 ∥ 风控) → 汇合
    builder.add_conditional_edges("collection_team", _fan_out_analysis_risk,
                                  ["analysis_team", "risk_team"])
    builder.add_edge("analysis_team", "synthesizer")
    builder.add_edge("risk_team", "synthesizer")

    # C 模式循环：planner → researcher×N → critic →(pass)→ synthesizer
    builder.add_conditional_edges("deep_planner", _fan_out_researchers, ["researcher"])
    builder.add_edge("researcher", "critic")
    builder.add_conditional_edges("critic", critic_gate,
                                  {"revise": "deep_planner", "pass": "synthesizer"})

    # 审批门控（human-in-the-loop）
    builder.add_edge("synthesizer", "human_review")
    builder.add_conditional_edges("human_review", human_gate,
                                  {"approve": "reporter", "revise": "synthesizer"})
    builder.add_edge("reporter", END)

    return builder.compile(checkpointer=checkpointer)
