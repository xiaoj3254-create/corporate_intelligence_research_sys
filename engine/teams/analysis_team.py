"""分析组：finance_agent / market_agent（依赖搜集组产出 prior_findings）。"""
from engine.prompts import WORKER_SYSTEM_PROMPTS
from engine.teams.common import WorkerSpec, build_team, wrap_team
from knowledge.hybrid_retriever import rag_retrieve
from tools.search import tavily_search


def build_analysis_team():
    return build_team("analysis", [
        WorkerSpec(
            name="finance_agent",
            task_types=["finance"],
            tools=[tavily_search, rag_retrieve],
            system_prompt=WORKER_SYSTEM_PROMPTS["finance_agent"]),
        WorkerSpec(
            name="market_agent",
            task_types=["market"],
            tools=[tavily_search, rag_retrieve],
            system_prompt=WORKER_SYSTEM_PROMPTS["market_agent"]),
    ])


analysis_team_node = wrap_team(build_analysis_team())
