"""联网搜索工具：Tavily + SQLite 缓存；无 API Key 时回退到确定性模拟结果（离线演示）。"""
import hashlib

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


@tool
def tavily_search(query: str, max_results: int = 5) -> str:
    """联网搜索工具：输入搜索关键词，返回搜索结果（标题、链接、摘要）。"""
    key = hashlib.sha256(f"{query}|{max_results}".encode("utf-8")).hexdigest()
    cached = db.cache_get(key)
    if cached:
        return cached

    if settings.tavily_api_key:
        try:
            from tavily import TavilyClient
            resp = TavilyClient(api_key=settings.tavily_api_key).search(
                query=query, max_results=max_results)
            results = resp.get("results", [])
            if results:
                text = "\n\n".join(
                    f"【{r.get('title', '')}】\n{r.get('url', '')}\n{r.get('content', '')}"
                    for r in results)
                db.cache_put(key, text)
                return text
        except Exception as e:
            # 降级结果不写缓存：一次瞬时网络故障若被固化，之后同一 query
            # 永远返回假数据（缓存无 TTL 会跨进程存活）。mock 是确定性
            # 生成的，重算无成本，没必要缓存。
            return f"(Tavily 搜索异常: {e}，已降级为模拟数据)\n\n" + _mock_search(query, max_results)

    # 无 key 的离线演示路径：确定性 mock，不写缓存（避免日后配置 key 后
    # 命中同 key 的旧 mock 结果）
    return _mock_search(query, max_results)
