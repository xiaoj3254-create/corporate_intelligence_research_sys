"""风控组：compliance_agent（依赖搜集组产出 prior_findings）。"""
from engine.prompts import WORKER_SYSTEM_PROMPTS
from engine.teams.common import WorkerSpec, build_team, wrap_team
from knowledge.hybrid_retriever import rag_retrieve
from tools.registry_mock import registry_lookup
from tools.search import tavily_search


def build_risk_team():
    return build_team("risk", [
        WorkerSpec(
            name="compliance_agent",
            task_types=["compliance"],
            tools=[tavily_search, registry_lookup, rag_retrieve],
            system_prompt=WORKER_SYSTEM_PROMPTS["compliance_agent"]),
    ])


risk_team_node = wrap_team(build_risk_team())
