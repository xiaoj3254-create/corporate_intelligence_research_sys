"""网页正文抓取工具：trafilatura 正文提取，失败时降级 requests，再失败返回模拟内容。"""
from langchain_core.tools import tool


@tool
def web_crawl(url: str) -> str:
    """网页正文抓取工具：输入 URL，返回页面正文文本。"""
    if "mock.example.com" in url or not url.startswith("http"):
        return (f"(模拟页面正文) 关于 {url} 的演示内容：该页面包含相关企业与行业信息……"
                f"（离线演示数据）")
    try:
        import trafilatura
        doc = trafilatura.fetch_url(url)
        if doc:
            text = trafilatura.extract(doc) or ""
            if text.strip():
                return text[:4000]
    except Exception:
        pass
    try:
        import requests
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        return r.text[:4000]
    except Exception as e:
        return f"(抓取失败: {e})"
