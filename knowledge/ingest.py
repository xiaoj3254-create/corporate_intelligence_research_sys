"""情报入库：调研产出 → FTS5(BM25) + Chroma(向量) 双写，沉淀为可复用的情报资产。

写入顺序刻意为 FTS 先行：BM25 不依赖 embedding API，永远可用；
向量侧未配置 EMBEDDING_API_KEY 或 API 失败时只降级告警，不连累 BM25。
chunk_id 为内容摘要：同一内容重跑不会重复沉淀（双写幂等）。
"""
import hashlib
import logging
import re

from knowledge.vector_store import get_collection
from storage import db

logger = logging.getLogger(__name__)


def _chunk_id(company: str, doc: str) -> str:
    return hashlib.md5(f"{company or ''}|{doc}".encode("utf-8")).hexdigest()


def _vector_upsert(collection_name: str, docs: list, metas: list, ids: list) -> bool:
    """向量写入；未配置 embedding key 或 API 失败时返回 False（不抛出）。"""
    try:
        col = get_collection(collection_name)
        col.upsert(documents=docs, metadatas=metas, ids=ids)
        return True
    except Exception as e:
        logger.warning("向量入库失败（BM25 不受影响）: %s", e)
        return False


def ingest_findings(company: str, task_id: str, findings) -> int:
    """每条 Finding 作为一个 chunk 双写 intel_store。"""
    if not findings:
        return 0
    docs = [f"[{f.team}/{f.agent}] {f.task}\n{f.content}" for f in findings]
    metas = [{
        "company": company,
        "team": f.team,
        "agent": f.agent,
        "credibility": f.credibility,
        "task_id": task_id,
        "source_urls": ", ".join(f.source_urls[:3]),
    } for f in findings]
    ids = [_chunk_id(company, d) for d in docs]

    for cid, doc in zip(ids, docs):
        db.fts_insert(doc, company, cid)
    _vector_upsert("intel_store", docs, metas, ids)
    return len(ids)


def ingest_report(company: str, task_id: str, markdown: str) -> int:
    """历史报告按 Markdown 二级标题切分后双写 reports_store。"""
    sections = [s.strip() for s in re.split(r"\n(?=## )", markdown) if s.strip()]
    if not sections:
        return 0
    metas = [{"company": company, "team": "report", "agent": "reporter",
              "task_id": task_id} for _ in sections]
    ids = [_chunk_id(company, s) for s in sections]

    for cid, doc in zip(ids, sections):
        db.fts_insert(doc, company, cid)
    _vector_upsert("reports_store", sections, metas, ids)
    return len(ids)
