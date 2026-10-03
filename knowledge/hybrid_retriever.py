"""混合检索：Chroma 向量 + SQLite FTS5(BM25) → RRF 融合。"""
from langchain_core.tools import tool

from knowledge.vector_store import get_collection
from storage import db


def rrf_fuse(rankings: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion：多路排序融合，返回融合后的 chunk_id 排序。"""
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]


def hybrid_retrieve(query: str, company: str = None, k: int = 5,
                    collection_name: str = "intel_store") -> list:
    """混合检索，返回 [{content, metadata}, ...]。"""
    col = get_collection(collection_name)
    where = {"company": company} if company else None

    # 召回 1: 向量检索
    vec_ids, vec_docs, vec_metas = [], [], []
    try:
        res = col.query(query_texts=[query], n_results=20, where=where)
        vec_ids = res.get("ids", [[]])[0] or []
        vec_docs = res.get("documents", [[]])[0] or []
        vec_metas = res.get("metadatas", [[]])[0] or []
    except Exception:
        pass

    id2doc = {}
    rankings = []
    if vec_ids:
        rankings.append(vec_ids)
        for i, cid in enumerate(vec_ids):
            id2doc[cid] = {"content": vec_docs[i], "metadata": vec_metas[i] or {}}

    # 召回 2: BM25 (FTS5)
    try:
        bm_rows = db.fts_search(query, company=company, limit=20)
    except Exception:
        bm_rows = []
    if bm_rows:
        bm_ids = [r["chunk_id"] for r in bm_rows]
        rankings.append(bm_ids)
        for r in bm_rows:
            id2doc.setdefault(r["chunk_id"],
                              {"content": r["content"], "metadata": {"source": "bm25"}})

    if not rankings:
        return []
    fused = rrf_fuse(rankings)
    return [id2doc[cid] for cid in fused[:k] if cid in id2doc]


@tool
def rag_retrieve(query: str, company: str = "", k: int = 5) -> str:
    """情报知识库检索工具：检索历史调研沉淀的情报与报告，优先于联网搜索使用。"""
    try:
        results = hybrid_retrieve(query, company=company or None, k=k)
    except Exception as e:
        return f"(知识库检索失败: {e})"
    if not results:
        return "(知识库中暂无相关历史情报)"
    return "\n\n".join(
        f"[{r['metadata'].get('team', 'report')}/{r['metadata'].get('agent', '')}] "
        f"{r['content'][:400]}"
        for r in results)
