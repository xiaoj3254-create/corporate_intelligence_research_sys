"""专利信息查询工具（演示用确定性模拟数据）。"""
import hashlib

from langchain_core.tools import tool


@tool
def patent_lookup(company: str) -> str:
    """专利信息查询工具：输入公司名称，返回专利布局概况。"""
    h = int(hashlib.md5(("patent:" + company).encode("utf-8")).hexdigest(), 16)
    fields = ["人工智能", "电池技术", "影像处理", "通信技术", "智能制造", "芯片设计"]
    n_patent = 200 + h % 3000
    return f"""「{company}」专利概况（演示用模拟数据）:
- 累计专利申请: {n_patent} 件，其中发明专利占比 {50 + h % 40}%
- 近三年新增: {n_patent // 5} 件
- 主要技术方向: {fields[h % len(fields)]}、{fields[(h + 1) % len(fields)]}
- 海外(PCT)布局占比: {h % 30}%"""
