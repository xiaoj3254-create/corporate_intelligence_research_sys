"""通用 ReAct Worker 构建器与执行封装。

性能说明: 实测 ~90% 耗时在 LLM 长上下文调用。两项治理措施：
1. compress_tool —— 工具输出截断（搜索/抓取动辄上万字，全文回灌会
   让每轮 ReAct 调用的上下文线性膨胀）；
2. stream_agent_trajectory —— ReAct 轮数上限，达限后保留已产出的部分轨迹
   继续生成 finding，不让单个 worker 无限循环拖慢整条流水线
   （团队 worker 与 deep 模式 researcher 共用此保护）。
"""
import re

from langgraph.errors import GraphRecursionError

from engine.config import get_llm, settings
from engine.prompts import WORKER_USER_PROMPT
from engine.state import Finding, now_str

_URL_RE = re.compile(r"https?://[^\s)\]】>,，]+")
# 离线演示数据里的假链接域名（tools/search.py 的 mock 结果）
_MOCK_URL_HOST = "mock.example.com"


def compress_tool(tool, max_chars: int):
    """把工具输出截断到 max_chars，保留原 name/description/args_schema。"""
    from langchain_core.tools import StructuredTool

    def _run(**kwargs):
        out = tool.invoke(kwargs)
        text = out if isinstance(out, str) else str(out)
        if len(text) > max_chars:
            text = (text[:max_chars]
                    + f"\n\n(输出已截断: 原文 {len(text)} 字，仅保留前 {max_chars} 字。"
                      "如需更多细节，请用更精确的参数再次调用)")
        return text

    return StructuredTool(
        name=tool.name, description=tool.description,
        args_schema=tool.args_schema, func=_run)


def make_worker_agent(name: str, tools: list, system_prompt: str):
    """基于 create_react_agent 一行构建专职 worker（每个 agent 只绑定 2-3 个相关工具，
    工具越多 ReAct 选错率越高）。工具统一做输出压缩。

    注意: create_react_agent 自 LangGraph V1.0 起弃用（V2.0 移除），
    迁移目标为 langchain.agents.create_agent。requirements.txt 已锁
    langgraph<2.0.0，迁移时同步解除版本上限。
    """
    from langgraph.prebuilt import create_react_agent

    max_chars = settings.tool_output_max_chars
    return create_react_agent(
        get_llm("strong"),
        tools=[compress_tool(t, max_chars) for t in tools],
        prompt=system_prompt)


def stream_agent_trajectory(agent, user: str):
    """执行 ReAct agent 并收集消息轨迹，返回 (msgs, hit_limit)。

    用 stream(values) 收集而非 invoke：触发 recursion_limit 时
    GraphRecursionError 之前已收集的消息仍然可用（部分产出不丢弃）。
    每轮 ReAct = agent + tools 两个 superstep；+2 留给最终回答。
    """
    config = {"recursion_limit": settings.max_worker_tool_calls * 2 + 2}
    msgs, hit_limit = [], False
    try:
        for snap in agent.stream({"messages": [("user", user)]}, config,
                                 stream_mode="values"):
            msgs = snap.get("messages") or msgs
    except Exception as e:
        # GraphRecursionError（达轮数上限）保留已收集的部分产出；其他异常上抛
        if isinstance(e, GraphRecursionError):
            hit_limit = True
        else:
            raise
    return msgs, hit_limit


def extract_urls(messages) -> list:
    """从完整消息轨迹（含工具调用参数与结果）中抽取引用 URL。

    mock.example.com 的假链接既不作为来源入库，也不参与可信度计数
    （否则离线模拟数据会被 URL 数量启发式评为 high 可信度）。
    """
    urls = []
    for m in messages:
        text = m.content if isinstance(m.content, str) else str(m.content or "")
        for u in _URL_RE.findall(text):
            if u in urls or _MOCK_URL_HOST in u:
                continue
            urls.append(u)
    return urls[:6]


def credibility_of(urls: list) -> str:
    """按独立来源数粗评可信度：≥2 high / 1 medium / 无来源 low。

    mock 链接已在 extract_urls 中剔除，离线模拟产出天然落入 low。
    """
    return "high" if len(urls) >= 2 else ("medium" if urls else "low")


def _final_text(msgs: list) -> str:
    """取轨迹中最后一条有文本的 AI 消息（中途达限时不至于丢失产出）。

    绝不回退到工具消息原文——那是原始工具输出而非分析结论，
    混入 finding 会污染知识库；找不到时返回空串，由 finding_text 给占位。
    """
    for m in reversed(msgs):
        if getattr(m, "type", "") != "ai":
            continue
        c = m.content
        if isinstance(c, list):  # 多模态内容块
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        if isinstance(c, str) and c.strip():
            return c
    return ""


def finding_text(msgs: list) -> str:
    """finding 正文：最后一条有文本的 AI 消息；全程无文本（达限被截断在
    工具执行中途）时用最后一次工具调用生成占位说明，而不是工具原文。"""
    text = _final_text(msgs)
    if text:
        return text
    for m in reversed(msgs):
        if getattr(m, "type", "") != "ai":
            continue
        calls = getattr(m, "tool_calls", None) or []
        if calls:
            last = calls[-1]
            args = ", ".join(f"{k}={str(v)[:80]}"
                             for k, v in (last.get("args") or {}).items())
            return f"(无文本产出; 最后动作: {last.get('name', '?')}({args}))"
    return "(无产出)"


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

    msgs, hit_limit = stream_agent_trajectory(agent, user)

    content = finding_text(msgs)
    urls = extract_urls(msgs)
    finding = Finding(team=team, agent=agent_name, task=task_desc,
                      content=content, source_urls=urls,
                      credibility=credibility_of(urls))

    n_tools = sum(1 for m in msgs
                  if getattr(m, "type", "") == "ai" and getattr(m, "tool_calls", None))
    suffix = "（达到工具调用轮数上限，产出为部分结果）" if hit_limit else ""
    event = {
        "node": agent_name, "agent": agent_name, "task": task_desc,
        "status": "done", "detail": f"工具调用 {n_tools} 次，产出 {len(content)} 字{suffix}",
        "ts": now_str(),
    }
    return finding, event
