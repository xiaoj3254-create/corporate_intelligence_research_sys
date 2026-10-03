"""深度模式并行研究员：每个实例只负责一个子问题（由 Send 动态 fan-out）。"""
from engine.agents.react_workers import extract_urls, make_worker_agent
from engine.prompts import RESEARCHER_PROMPT, WORKER_SYSTEM_PROMPTS
from engine.state import Finding, now_str

_agent = None


def _get_agent():
    global _agent
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
    """Send 入参: {company, objective, sub_q: SubQuestion}。"""
    sq = state["sub_q"]
    question = sq.question if hasattr(sq, "question") else str(sq)

    agent = _get_agent()
    user = RESEARCHER_PROMPT.format(
        company=state["company"],
        objective=state.get("objective", ""),
        sub_question=question)
    result = agent.invoke({"messages": [("user", user)]})
    msgs = result["messages"]
    content = msgs[-1].content or "(无产出)"
    if not isinstance(content, str):
        content = str(content)

    urls = extract_urls(msgs)
    finding = Finding(team="deep", agent="researcher", task=question,
                      content=content, source_urls=urls,
                      credibility="high" if len(urls) >= 2 else "medium")
    n_tools = sum(1 for m in msgs
                  if getattr(m, "type", "") == "ai" and getattr(m, "tool_calls", None))
    return {
        "findings": [finding],
        "timeline": [{
            "node": "researcher", "agent": "researcher", "task": question,
            "status": "done", "detail": f"工具调用 {n_tools} 次", "ts": now_str(),
        }],
    }
