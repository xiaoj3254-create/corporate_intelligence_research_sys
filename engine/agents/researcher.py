"""深度模式并行研究员：每个实例只负责一个子问题（由 Send 动态 fan-out）。"""
import threading

from engine.agents.react_workers import (
    credibility_of,
    extract_urls,
    finding_text,
    make_worker_agent,
    stream_agent_trajectory,
)
from engine.events import emit_live
from engine.prompts import RESEARCHER_PROMPT, WORKER_SYSTEM_PROMPTS
from engine.state import Finding, now_str

_agent = None
_agent_lock = threading.Lock()


def _get_agent():
    global _agent
    if _agent is None:
        # 双检锁: Send 动态 fan-out 下多个 researcher 可能同时首次触发构建
        with _agent_lock:
            if _agent is None:
                from knowledge.hybrid_retriever import rag_retrieve
                from tools.crawler import web_crawl
                from tools.search import tavily_search
                _agent = make_worker_agent(
                    "researcher",
                    [tavily_search, web_crawl, rag_retrieve],
                    WORKER_SYSTEM_PROMPTS["researcher"])
    return _agent


def researcher_node(state: dict) -> dict:
    """Send 入参: {company, objective, sub_q, task_id}。"""
    sq = state["sub_q"]
    question = sq.question if hasattr(sq, "question") else str(sq)

    agent = _get_agent()
    user = RESEARCHER_PROMPT.format(
        company=state["company"],
        objective=state.get("objective", ""),
        sub_question=question)

    # 与 run_worker 同款递归保护：达轮数上限保留部分产出，不炸整图
    msgs, hit_limit = stream_agent_trajectory(agent, user)

    content = finding_text(msgs)
    urls = extract_urls(msgs)
    finding = Finding(team="deep", agent="researcher", task=question,
                      content=content, source_urls=urls,
                      credibility=credibility_of(urls))

    n_tools = sum(1 for m in msgs
                  if getattr(m, "type", "") == "ai" and getattr(m, "tool_calls", None))
    suffix = "（达到工具调用轮数上限，产出为部分结果）" if hit_limit else ""
    event = {
        "node": "researcher", "agent": "researcher", "task": question,
        "status": "done", "detail": f"工具调用 {n_tools} 次，产出 {len(content)} 字{suffix}",
        "ts": now_str(),
    }
    # 实时外推: Send 并行下主图 stream 要等本 superstep 里最慢的 researcher
    # 结束才统一吐出，逐实例外推可让先完成者的事件即刻到达前端
    emit_live(state.get("task_id", ""), event)
    return {"findings": [finding], "timeline": [event]}
