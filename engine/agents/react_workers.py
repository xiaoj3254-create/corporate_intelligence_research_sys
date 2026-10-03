"""通用 ReAct Worker 构建器与执行封装。"""
import re

from engine.config import get_llm
from engine.prompts import WORKER_USER_PROMPT
from engine.state import Finding, now_str

_URL_RE = re.compile(r"https?://[^\s)\]】>,，]+")


def make_worker_agent(name: str, tools: list, system_prompt: str):
    """基于 create_react_agent 一行构建专职 worker（每个 agent 只绑定 2-3 个相关工具，
    工具越多 ReAct 选错率越高）。

    注意: create_react_agent 自 LangGraph V1.0 起弃用（V2.0 移除），
    迁移目标为 langchain.agents.create_agent。requirements.txt 已锁
    langgraph<2.0.0，迁移时同步解除版本上限。
    """
    from langgraph.prebuilt import create_react_agent
    return create_react_agent(get_llm("strong"), tools=tools, prompt=system_prompt)


def extract_urls(messages) -> list:
    """从完整消息轨迹（含工具调用参数与结果）中抽取引用 URL。"""
    urls = []
    for m in messages:
        text = m.content if isinstance(m.content, str) else str(m.content or "")
        for u in _URL_RE.findall(text):
            if u not in urls:
                urls.append(u)
    return urls[:6]


def run_worker(agent, agent_name: str, team: str, task_desc: str,
               company: str, objective: str, prior_findings: list):
    """执行单个 worker，返回 (Finding, timeline 事件)。"""
    prior_section = ""
    if prior_findings:
        prior_section = "可参考的上游情报:\n" + "\n".join(
            f"- [{f.agent}] {f.content[:200]}" for f in prior_findings[:8])

    user = WORKER_USER_PROMPT.format(
        company=company, objective=objective or "全面调研",
        task=task_desc, prior_section=prior_section)

    result = agent.invoke({"messages": [("user", user)]})
    msgs = result["messages"]
    content = msgs[-1].content or "(无产出)"
    if not isinstance(content, str):
        content = str(content)

    urls = extract_urls(msgs)
    credibility = "high" if len(urls) >= 2 else ("medium" if urls else "low")
    finding = Finding(team=team, agent=agent_name, task=task_desc,
                      content=content, source_urls=urls, credibility=credibility)

    n_tools = sum(1 for m in msgs
                  if getattr(m, "type", "") == "ai" and getattr(m, "tool_calls", None))
    event = {
        "node": agent_name, "agent": agent_name, "task": task_desc,
        "status": "done", "detail": f"工具调用 {n_tools} 次，产出 {len(content)} 字",
        "ts": now_str(),
    }
    return finding, event
