"""联网搜索工具：三级降级链 Tavily → Firecrawl → 确定性模拟；SQLite 缓存。

降级语义:
- Tavily 主通道；异常或返回空结果时启用 Firecrawl（未配置 key 则跳过该级）
- 两级都失败才降级模拟数据（确定性 mock，便于离线演示与测试）
- 成功结果（任一真实通道）写缓存；mock 永不写缓存——瞬时故障若被固化，
  之后同一 query 在缓存 TTL 内永远返回假数据
- 通道来源写入结果首行标记：组长质检与 synthesizer 的可信度评级依赖该信息
"""
import hashlib

import requests
from langchain_core.tools import tool

from engine.config import settings
from storage import db


def _mock_search(query: str, n: int) -> str:
    """确定性模拟搜索结果（同一 query 永远返回同一结果，便于演示与测试）。"""
    h = int(hashlib.md5(query.encode("utf-8")).hexdigest(), 16)
    items = []
    for i in range(min(n, 3)):
        seed = h % 1000 + i * 37
        items.append(
            f"【{query} - 公开报道 {i + 1}】(模拟数据)\n"
            f"https://mock.example.com/{ hashlib.md5(f'{query}{i}'.encode()).hexdigest()[:10] }/{seed}\n"
            f"关于「{query}」的模拟搜索摘要 {seed}：该主题近期受到行业关注，多家媒体报道了相关动态，"
            f"整体呈现稳健发展趋势。（离线演示数据，配置 TAVILY_API_KEY 后启用真实搜索）")
    return "\n\n".join(items)


def _tavily_search(query: str, n: int) -> str:
    """Tavily 主通道；返回结果文本，空字符串表示无结果，异常向上抛。"""
    from tavily import TavilyClient
    resp = TavilyClient(api_key=settings.tavily_api_key).search(
        query=query, max_results=n)
    results = resp.get("results", [])
    if not results:
        return ""
    return "\n\n".join(
        f"【{r.get('title', '')}】\n{r.get('url', '')}\n{r.get('content', '')}"
        for r in results)


def _firecrawl_search(query: str, n: int, timeout: int = 15) -> str:
    """Firecrawl 降级通道（/search 含正文提取）；约定同上。短超时快速失败，防拖慢 ReAct 轮。"""
    resp = requests.post(
        f"{settings.firecrawl_base_url.rstrip('/')}/search",
        headers={"Authorization": f"Bearer {settings.firecrawl_api_key}",
                 "Content-Type": "application/json"},
        json={"query": query, "limit": n},
        timeout=timeout)
    resp.raise_for_status()
    # 兼容两种响应结构: v1 data 直接是结果列表；v2 是 {"web": [...]}
    data = resp.json().get("data")
    if isinstance(data, list):
        web = data
    elif isinstance(data, dict):
        web = data.get("web") or []
    else:
        web = []
    if not web:
        return ""
    items = []
    for r in web[:n]:
        body = r.get("markdown") or r.get("description") or ""
        items.append(f"【{r.get('title', '')}】\n{r.get('url', '')}\n{body[:1500]}")
    return "\n\n".join(items)


@tool
def tavily_search(query: str, max_results: int = 5) -> str:
    """联网搜索工具：输入搜索关键词，返回搜索结果（标题、链接、摘要）。"""
    key = hashlib.sha256(f"{query}|{max_results}".encode("utf-8")).hexdigest()
    cached = db.cache_get(key)
    if cached:
        return cached

    attempts = []
    # 1) Tavily 主通道
    if settings.tavily_api_key:
        try:
            text = _tavily_search(query, max_results)
            if text:
                text = f"(搜索通道: tavily)\n\n{text}"
                db.cache_put(key, text)
                return text
            attempts.append("tavily: 无结果")
        except Exception as e:
            attempts.append(f"tavily: {e}")

    # 2) Firecrawl 降级通道（未配置 key 则跳过该级）
    if settings.firecrawl_api_key:
        try:
            text = _firecrawl_search(query, max_results)
            if text:
                text = f"(搜索通道: firecrawl 降级)\n\n{text}"
                db.cache_put(key, text)
                return text
            attempts.append("firecrawl: 无结果")
        except Exception as e:
            attempts.append(f"firecrawl: {e}")

    # 3) 确定性 mock 兜底（不写缓存，避免固化瞬时故障）
    note = "；".join(attempts) if attempts else "未配置任何搜索 API key"
    return f"(搜索不可用已降级模拟数据: {note})\n\n" + _mock_search(query, max_results)
