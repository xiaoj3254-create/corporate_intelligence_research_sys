"""团队子图工厂：组长拆解 → Send 组内并行 → 组长质检（不足则组内修订）。"""
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from engine.agents.react_workers import make_worker_agent, run_worker
from engine.config import get_llm, settings
from engine.prompts import TEAM_LEAD_DECOMPOSE_PROMPTS, TEAM_LEAD_REVIEW_PROMPT
from engine.state import ReviewResult, Task, TaskList, TeamState, now_str


@dataclass
class WorkerSpec:
    name: str           # 节点名
    task_types: list    # 可承接的任务类型
    tools: list
    system_prompt: str


def _ev(node, task, detail, status="done"):
    return {"node": node, "agent": node, "task": task,
            "status": status, "detail": detail, "ts": now_str()}


# ---------------------------------------------------------------- 组长拆解

def _make_lead_decompose(team: str, allowed_types: list):
    def node(state: dict) -> dict:
        # 修订轮（lead_review 质检不通过后回来）：直接执行 lead_review 筛出的
        # missing 任务，不再调用 LLM 全量重拆——否则会把 missing 清单覆盖掉
        if state.get("revise_count"):
            tasks = state.get("tasks") or []
            return {"tasks": tasks,
                    "timeline": [_ev(f"{team}_lead", "补充执行",
                                     f"修订轮：补做 {len(tasks)} 个任务")]}
        prior = state.get("prior_findings") or []
        if prior:
            prior_section = "上游已搜集情报（摘要）:\n" + "\n".join(
                f"- [{f.agent}] {f.content[:160]}" for f in prior[:10])
        else:
            prior_section = "暂无上游情报。"

        prompt = TEAM_LEAD_DECOMPOSE_PROMPTS[team].format(
            company=state["company"],
            objective=state.get("objective", ""),
            prior_section=prior_section)

        tasks = []
        try:
            out = get_llm("cheap").with_structured_output(TaskList).invoke(prompt)
            tasks = [t for t in (out.tasks or []) if t.type in allowed_types]
        except Exception:
            tasks = []
        if not tasks:
            tasks = [Task(id="t-fallback", type=allowed_types[0],
                          description=f"调研 {state['company']} 的基本信息")]
        tasks = tasks[:settings.max_workers_tasks]
        return {"tasks": tasks,
                "timeline": [_ev(f"{team}_lead", "任务拆解", f"{len(tasks)} 个任务")]}
    return node


# ---------------------------------------------------------------- 组内 worker

_worker_agents: dict = {}


def _get_worker_agent(spec: WorkerSpec):
    """惰性构建 worker agent：支持测试时先注入 FakeChatModel 再跑图。"""
    if spec.name not in _worker_agents:
        _worker_agents[spec.name] = make_worker_agent(
            spec.name, spec.tools, spec.system_prompt)
    return _worker_agents[spec.name]


def _make_worker_node(spec: WorkerSpec, team: str):
    def node(state: dict) -> dict:
        agent = _get_worker_agent(spec)
        task = state.get("task")
        desc = task.description if hasattr(task, "description") else str(task)
        finding, event = run_worker(
            agent, spec.name, team, desc,
            state["company"], state.get("objective", ""),
            state.get("prior_findings") or [])
        return {"findings": [finding], "timeline": [event]}
    return node


def _make_assign(type_to_node: dict):
    """组内 Send 并行：按任务类型把任务分发给对应 worker。"""
    def assign(state: dict):
        sends = []
        for t in state.get("tasks") or []:
            target = type_to_node.get(t.type)
            if target:
                sends.append(Send(target, {
                    "company": state["company"],
                    "objective": state.get("objective", ""),
                    "task": t,
                    "prior_findings": state.get("prior_findings") or [],
                }))
        return sends
    return assign


# ---------------------------------------------------------------- 组长质检

def _make_lead_review(team: str, allowed_types: list):
    def node(state: dict) -> dict:
        tasks = state.get("tasks") or []
        findings = state.get("findings") or []
        revise = state.get("revise_count") or 0

        ok, missing = bool(findings) and len(findings) >= len(tasks), []
        try:
            out = get_llm("cheap").with_structured_output(ReviewResult).invoke(
                TEAM_LEAD_REVIEW_PROMPT.format(
                    team_name=team,
                    tasks="\n".join(f"- [{t.type}] {t.description}" for t in tasks) or "(无)",
                    findings="\n\n".join(
                        f"[{f.agent}] {f.task}\n{f.content[:400]}" for f in findings) or "(无)"))
            ok, missing = bool(out.pass_ok), list(out.missing_tasks or [])
        except Exception:
            pass
        # 只补做本团队 worker 能承接的类型，否则修订轮会空转到上限
        missing = [t for t in missing if t.type in allowed_types]

        update = {"timeline": [_ev(f"{team}_lead", "产出质检",
                                   "通过" if ok else "不通过")]}
        if ok:
            update["quality_ok"] = True
        elif revise < settings.max_team_revise and missing:
            # 组内修订：只补做 missing 任务
            update["quality_ok"] = False
            update["tasks"] = missing[:settings.max_workers_tasks]
            update["revise_count"] = revise + 1
        else:
            update["quality_ok"] = True  # 达到修订上限，放行
        return update
    return node


def _review_gate(state: dict):
    return END if state.get("quality_ok") else "lead_decompose"


# ---------------------------------------------------------------- 组装

def build_team(team: str, workers: list):
    """构建一个团队子图。"""
    builder = StateGraph(TeamState)
    allowed = sorted({t for w in workers for t in w.task_types})

    type_to_node = {}
    builder.add_node("lead_decompose", _make_lead_decompose(team, allowed))
    for spec in workers:
        builder.add_node(spec.name, _make_worker_node(spec, team))
        for t in spec.task_types:
            type_to_node[t] = spec.name
    builder.add_node("lead_review", _make_lead_review(team, allowed))

    builder.add_edge(START, "lead_decompose")
    builder.add_conditional_edges("lead_decompose", _make_assign(type_to_node),
                                  [w.name for w in workers])
    for spec in workers:
        builder.add_edge(spec.name, "lead_review")
    builder.add_conditional_edges("lead_review", _review_gate, ["lead_decompose", END])
    return builder.compile()


def wrap_team(subgraph):
    """把子图包装为主图节点：只回传 findings / timeline。

    避免子图内部字段（tasks/quality_ok/...）污染父图状态；
    findings/timeline 经父图同名字段 + add_list reducer 自动合并。
    """
    def node(state: dict) -> dict:
        result = subgraph.invoke(dict(state))
        out = {}
        if result.get("findings"):
            out["findings"] = result["findings"]
        if result.get("timeline"):
            out["timeline"] = result["timeline"]
        return out
    return node
