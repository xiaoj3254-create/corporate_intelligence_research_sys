"""零成本端到端拓扑测试：用 FakeChatModel 跑通 standard / deep 两种模式全图，
验证 Send 并行、条件门控、子图状态映射与审批链路（不调用真实 LLM）。

原理: FakeAgentChatModel
- 普通对话返回固定文本 → worker / synthesizer 走通
- with_structured_output 抛异常 → planner / critic / 组长全部走内置 fallback
  （各节点均做了降级设计），从而在无 LLM 条件下验证图的完整拓扑。
"""
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

FAKE_TEXT = "模拟调研产出：该企业基本面稳健，未发现重大风险信号。来源: https://mock.example.com/demo"


def _inf(text):
    while True:
        yield text


class FakeAgentChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self  # 让 create_react_agent 正常构建

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError("fake model: 走节点 fallback 路径")


@pytest.fixture
def fake_llm_env(tmp_path, monkeypatch):
    import engine.config as config
    import knowledge.hybrid_retriever as hr
    import knowledge.ingest as ing
    from storage import db

    # 1) 隔离数据目录（SQLite / Chroma 全部落到 tmp_path）
    monkeypatch.setattr(config.settings, "data_dir", str(tmp_path))
    db._conn = None

    # 2) 注入假模型（strong / cheap 双档）
    saved = dict(config._llm_cache)
    fake = FakeAgentChatModel(messages=_inf(FAKE_TEXT))
    config._llm_cache["strong"] = fake
    config._llm_cache["cheap"] = fake

    # 3) RAG 隔离（避免测试时下载 embedding 模型）
    monkeypatch.setattr(hr, "hybrid_retrieve", lambda *a, **k: [])
    monkeypatch.setattr(ing, "ingest_findings", lambda *a, **k: 0)
    monkeypatch.setattr(ing, "ingest_report", lambda *a, **k: 0)

    yield

    config._llm_cache.clear()
    config._llm_cache.update(saved)
    db._conn = None


def _run(graph, mode):
    inputs = {
        "company": "测试公司", "objective": "全面调研", "mode": mode,
        "task_id": "smoke-task", "thread_id": "smoke-thread",
        "auto_approve": True,
        "plan": [], "findings": [], "timeline": [], "research_round": 0,
    }
    final = {}
    for chunk in graph.stream(inputs, {"recursion_limit": 60},
                              stream_mode="values"):
        final = chunk
    return final


def test_standard_mode_end_to_end(fake_llm_env):
    from engine.graph import build_graph
    final = _run(build_graph(), "standard")

    # 报告生成
    assert final.get("final_report")
    # 三个团队子图全部产出（Send 固定 fan-out + 子图状态映射回父图）
    teams = {f.team for f in final["findings"]}
    assert teams == {"collection", "analysis", "risk"}
    # 标准模式不进入深度循环
    assert final.get("research_round", 0) == 0
    # 时间线有完整链路事件
    nodes = [e["node"] for e in final["timeline"]]
    assert "intake" in nodes and "reporter" in nodes
    # 报告落库
    from storage import db
    assert db.get_report("smoke-task")


def test_deep_mode_end_to_end(fake_llm_env):
    from engine.graph import build_graph
    final = _run(build_graph(), "deep")

    assert final.get("final_report")
    # 深度模式产出来自 researcher（Send 动态 fan-out）
    teams = {f.team for f in final["findings"]}
    assert teams == {"deep"}
    # 至少完成一轮 plan → research → critic 循环
    assert final.get("research_round", 0) >= 1
    # critic 给出质检结论（fallback 放行）
    assert final.get("critic_verdict") is not None
