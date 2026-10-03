"""Chroma 向量库封装：intel_store（情报）/ reports_store（历史报告）两个 collection。

Embedding 使用 OpenAI 兼容的第三方 API（默认硅基流动 bge-m3，中文效果好），
不再依赖 Chroma 内置的本地 ONNX 模型（避免首次下载 80MB 且对中文效果差）。
未配置 EMBEDDING_API_KEY 时，EF 调用直接抛错，由上层（hybrid_retriever）
降级为纯 BM25 检索。
"""
import os

import requests

from engine.config import settings

_client = None

# 两个 collection 的用途:
#   intel_store    —— 所有 Finding（调研发现）
#   reports_store  —— 历史最终报告（按章节切分）
COLLECTIONS = ("intel_store", "reports_store")


class RemoteEmbeddingFunction:
    """OpenAI 兼容 /embeddings 接口的 Chroma EmbeddingFunction。

    任何提供 /v1/embeddings 的服务都可用（硅基流动 / OneAPI / vLLM 等）。
    """

    def name(self) -> str:
        return "remote_openai"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict):
        return RemoteEmbeddingFunction()

    def __call__(self, input):
        if not settings.embedding_api_key:
            raise RuntimeError(
                "未配置 EMBEDDING_API_KEY，向量检索不可用（BM25 兜底中）")
        resp = requests.post(
            f"{settings.embedding_base_url.rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
            json={"model": settings.embedding_model, "input": list(input)},
            timeout=30)
        resp.raise_for_status()
        data = resp.json()["data"]
        # 按 index 还原顺序，保证与输入一一对应
        return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]


_embedding_fn = None


def get_embedding_function():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = RemoteEmbeddingFunction()
    return _embedding_fn


def _get_client():
    global _client
    if _client is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        path = os.path.join(settings.data_dir, "chroma")
        os.makedirs(path, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=path, settings=ChromaSettings(anonymized_telemetry=False))
    return _client


def get_collection(name: str):
    if name not in COLLECTIONS:
        raise ValueError(f"未知 collection: {name}")
    return _get_client().get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"},
        embedding_function=get_embedding_function())
