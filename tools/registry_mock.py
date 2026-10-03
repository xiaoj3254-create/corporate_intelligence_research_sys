"""工商信息查询工具（演示用确定性模拟数据：同一公司名永远返回同一结果）。"""
import hashlib

from langchain_core.tools import tool


@tool
def registry_lookup(company: str) -> str:
    """工商信息查询工具：输入公司名称，返回注册信息、股东结构与经营范围等。"""
    h = int(hashlib.md5(company.encode("utf-8")).hexdigest(), 16)
    provinces = ["北京", "上海", "深圳", "杭州", "广州", "苏州"]
    prov = provinces[h % len(provinces)]
    year = 1984 + h % 40
    capital = (h % 900 + 100) * 10
    return f"""「{company}」工商信息（演示用模拟数据）:
- 统一社会信用代码: 91110{h % 10000:04d}MA{h % 100:02d}XK
- 成立年份: {year} 年
- 注册资本: {capital} 万元
- 登记机关: {prov}市市场监督管理局
- 企业状态: 存续
- 股东结构: 创始团队持股 {35 + h % 30}%，机构投资者持股 {10 + h % 20}%
- 经营范围: 技术开发、电子产品销售、软件开发、进出口业务等
- 分支机构: {h % 20 + 1} 家"""
