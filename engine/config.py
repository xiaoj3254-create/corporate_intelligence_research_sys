"""全局配置与 LLM 路由（strong / cheap 双档）。

LLM 采用 OpenAI 兼容的通用调用方式（ChatOpenAI），
可对接任何 OpenAI 兼容服务: DeepSeek 官方 / 硅基流动 / OneAPI / vLLM 等。
"""
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM（OpenAI 兼容；兼容旧变量名 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL）
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "DEEPSEEK_API_KEY"))
    openai_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        validation_alias=AliasChoices("OPENAI_BASE_URL", "DEEPSEEK_BASE_URL"))
    strong_model: str = "deepseek-chat"
    cheap_model: str = "deepseek-chat"

    # 搜索
    tavily_api_key: str = ""

    # Embedding（OpenAI 兼容的第三方 API，用于 Chroma 向量检索）
    # 示例（硅基流动，bge-m3 免费）: EMBEDDING_API_KEY=sk-xxx
    # 未配置时向量召回自动跳过，BM25 检索兜底
    embedding_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("EMBEDDING_API_KEY", "SILICONFLOW_API_KEY"))
    embedding_base_url: str = Field(
        default="https://api.siliconflow.cn/v1",
        validation_alias=AliasChoices("EMBEDDING_BASE_URL", "SILICONFLOW_BASE_URL"))
    embedding_model: str = "BAAI/bge-m3"

    # 编排参数
    max_research_rounds: int = 3      # 深度模式反思循环上限
    max_team_revise: int = 2          # 团队子图内部修订上限
    max_workers_tasks: int = 4        # 组长单次拆解任务数上限
    recursion_limit: int = 80         # LangGraph 递归上限

    # 存储
    data_dir: str = "data"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

_llm_cache: dict = {}


def get_llm(role: str = "strong"):
    """按角色获取 LLM：strong=规划/质检/综合，cheap=组长/摘要等轻任务。"""
    from langchain_openai import ChatOpenAI

    if role not in ("strong", "cheap"):
        role = "strong"
    if role not in _llm_cache:
        model = settings.strong_model if role == "strong" else settings.cheap_model
        _llm_cache[role] = ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key or "sk-not-set",
            base_url=settings.openai_base_url,
            temperature=0.2,
        )
    return _llm_cache[role]
