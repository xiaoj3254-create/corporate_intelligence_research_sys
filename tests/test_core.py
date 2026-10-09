"""核心逻辑单元测试（不调用真实 LLM，零成本运行）。"""
import pytest

from engine.state import QualityAssessment, add_list


# ---------------------------------------------------------------- reducer

def test_add_list_merges():
    assert add_list(["a"], ["b"]) == ["a", "b"]


def test_add_list_tolerates_none():
    assert add_list(None, ["a"]) == ["a"]
    assert add_list(["a"], None) == ["a"]
    assert add_list(None, None) == []


# ---------------------------------------------------------------- 反思门控

def test_critic_gate_pass():
    from engine.nodes.critic import critic_gate
    state = {"critic_verdict": QualityAssessment(overall_pass=True),
             "research_round": 1}
    assert critic_gate(state) == "pass"
    assert critic_gate({}) == "pass"


def test_critic_gate_revise_on_gaps():
    from engine.nodes.critic import critic_gate
    state = {"critic_verdict": QualityAssessment(
        overall_pass=False, coverage_gaps=["缺少财务数据"]),
        "research_round": 1}
    assert critic_gate(state) == "revise"


def test_critic_gate_pass_at_round_limit():
    from engine.config import settings
    from engine.nodes.critic import critic_gate
    state = {"critic_verdict": QualityAssessment(
        overall_pass=False, coverage_gaps=["仍未覆盖"]),
        "research_round": settings.max_research_rounds}
    assert critic_gate(state) == "pass"


# ---------------------------------------------------------------- 审批门控

def test_human_gate():
    from engine.nodes.human_review import human_gate
    assert human_gate({"human_feedback": "approve"}) == "approve"
    assert human_gate({"human_feedback": "补充供应链分析"}) == "revise"
    assert human_gate({}) == "revise"


# ---------------------------------------------------------------- RRF 融合

def test_rrf_fuse_ranks_consensus_first():
    from knowledge.hybrid_retriever import rrf_fuse
    rankings = [
        ["a", "b", "c", "d"],
        ["b", "a", "e"],
    ]
    fused = rrf_fuse(rankings)
    # a、b 两路都召回，应排在只有单路召回的 c/e 之前
    assert set(fused[:2]) == {"a", "b"}
    assert fused.index("a") < fused.index("c")
    assert fused.index("b") < fused.index("e")


def test_rrf_fuse_empty():
    from knowledge.hybrid_retriever import rrf_fuse
    assert rrf_fuse([]) == []
    assert rrf_fuse([[]]) == []


# ---------------------------------------------------------------- Mock 工具确定性

def test_registry_mock_deterministic():
    from tools.registry_mock import registry_lookup
    a = registry_lookup.invoke({"company": "小米"})
    b = registry_lookup.invoke({"company": "小米"})
    assert a == b
    assert "小米" in a
    assert "工商信息" in a


def test_patent_mock_deterministic():
    from tools.patent_mock import patent_lookup
    assert patent_lookup.invoke({"company": "宁德时代"}) == \
        patent_lookup.invoke({"company": "宁德时代"})


def test_mock_search_has_url():
    from tools.search import _mock_search
    text = _mock_search("小米 汽车", 3)
    assert "https://" in text
    assert "小米" in text


# ---------------------------------------------------------------- SQLite DAO

def test_db_roundtrip(tmp_path, monkeypatch):
    from storage import db
    monkeypatch.setattr(db.settings, "data_dir", str(tmp_path))
    db._conn = None

    db.create_task("t1", "th1", "小米", "评估合作", "deep")
    task = db.get_task("t1")
    assert task["company"] == "小米"
    assert task["mode"] == "deep"

    db.set_task_status("t1", "completed")
    assert db.get_task("t1")["status"] == "completed"

    db.save_event("t1", {"node": "intake", "agent": "router", "task": "任务解析",
                         "status": "done", "detail": "模式: deep", "ts": "12:00:00"})
    events = db.list_events("t1")
    assert len(events) == 1 and events[0]["node"] == "intake"

    db.cache_put("k1", "result")
    assert db.cache_get("k1") == "result"

    db.fts_insert("小米的专利布局分析", "小米", "chunk-1")
    rows = db.fts_search("专利布局", company="小米")
    assert len(rows) == 1 and rows[0]["chunk_id"] == "chunk-1"

    db.save_report("t1", "# 报告")
    assert db.get_report("t1") == "# 报告"

    db._conn = None  # 复位单例，避免污染其他测试


# ---------------------------------------------------------------- 图拓扑

def test_graph_compiles_with_expected_nodes():
    from engine.graph import build_graph
    g = build_graph()
    names = set(g.get_graph().nodes.keys())
    expected = {"intake", "collection_team", "analysis_team", "risk_team",
                "deep_planner", "researcher", "critic", "synthesizer",
                "human_review", "reporter"}
    assert expected <= names, f"缺少节点: {expected - names}"


# ---------------------------------------------------------------- worker 执行封装

class _Msg:
    """最小消息桩：只需 type / content / tool_calls 三个属性。"""

    def __init__(self, type, content=None, tool_calls=None):
        self.type = type
        self.content = content
        self.tool_calls = tool_calls


def test_extract_urls_filters_mock_domain():
    """mock.example.com 假链接不得作为来源（否则模拟数据会被评为高可信）。"""
    from engine.agents.react_workers import extract_urls
    msgs = [_Msg("tool", "https://mock.example.com/a1 "
                          "https://real.example.com/a https://real.example.com/b")]
    assert extract_urls(msgs) == ["https://real.example.com/a",
                                  "https://real.example.com/b"]


def test_credibility_of_url_count():
    from engine.agents.react_workers import credibility_of
    assert credibility_of(["u1", "u2"]) == "high"
    assert credibility_of(["u1"]) == "medium"
    assert credibility_of([]) == "low"


def test_finding_text_never_returns_tool_output():
    """无文本产出时不得把工具原文当 finding 正文，应用最后动作做占位。"""
    from engine.agents.react_workers import finding_text
    msgs = [_Msg("ai", tool_calls=[{"name": "tavily_search",
                                    "args": {"query": "小米 专利"}}]),
            _Msg("tool", "原始工具输出" * 200)]
    text = finding_text(msgs)
    assert "原始工具输出" not in text
    assert "tavily_search" in text


def test_researcher_survives_recursion_limit(monkeypatch):
    """P1 回归：deep 模式 researcher 达递归上限不得炸整图，保留占位产出。"""
    from langgraph.errors import GraphRecursionError

    import engine.agents.researcher as mod
    from engine.state import SubQuestion

    class ExplodingAgent:
        def stream(self, *a, **k):
            raise GraphRecursionError("limit")

    # 注入假 agent：否则 researcher_node 内部 _get_agent() 会构建真实 agent
    # 并发起真实 LLM 调用（慢、消耗配额、结果不可断言）
    monkeypatch.setattr(mod, "_agent", ExplodingAgent())
    state = {"company": "小米", "objective": "",
             "sub_q": SubQuestion(question="小米的主营业务"), "task_id": ""}
    out = mod.researcher_node(state)
    finding = out["findings"][0]
    assert finding.content
    assert finding.credibility == "low"
    assert "上限" in out["timeline"][0]["detail"]


def test_eval_tasks_hidden_from_list(tmp_path, monkeypatch):
    """eval 评估任务与正式任务同表存储，但 list_tasks 只返回正式任务。"""
    from storage import db
    monkeypatch.setattr(db.settings, "data_dir", str(tmp_path))
    db._conn = None
    db.create_task("t1", "th1", "小米", "评估合作", "deep")
    db.create_task("eval1", "th2", "小米", "评估集任务", "deep", kind="eval")
    ids = [t["id"] for t in db.list_tasks()]
    assert "t1" in ids and "eval1" not in ids
    assert db.get_task("eval1")["kind"] == "eval"
    db._conn = None  # 复位单例，避免污染其他测试
