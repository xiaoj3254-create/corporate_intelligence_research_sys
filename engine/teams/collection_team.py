"""搜集组：search_agent / registry_agent / patent_agent 组内 Send 并行。"""
from engine.prompts import WORKER_SYSTEM_PROMPTS
from engine.teams.common import WorkerSpec, build_team, wrap_team
from knowledge.hybrid_retriever import rag_retrieve
from tools.crawler import web_crawl
from tools.patent_mock import patent_lookup
from tools.registry_mock import registry_lookup
from tools.search import tavily_search


def build_collection_team():
    return build_team("collection", [
        WorkerSpec(
            name="search_agent",
            task_types=["search"],
            tools=[tavily_search, web_crawl, rag_retrieve],
            system_prompt=WORKER_SYSTEM_PROMPTS["search_agent"]),
        WorkerSpec(
            name="registry_agent",
            task_types=["registry"],
            tools=[registry_lookup, rag_retrieve],
            system_prompt=WORKER_SYSTEM_PROMPTS["registry_agent"]),
        WorkerSpec(
            name="patent_agent",
            task_types=["patent"],
            tools=[patent_lookup, rag_retrieve],
            system_prompt=WORKER_SYSTEM_PROMPTS["patent_agent"]),
    ])


collection_team_node = wrap_team(build_collection_team())
