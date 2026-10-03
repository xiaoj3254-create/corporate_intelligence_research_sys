"""reporter: 报告归档 —— SQLite 落库 + RAG 知识库沉淀（情报资产化）。"""
import logging

from engine.state import now_str
from storage import db

logger = logging.getLogger(__name__)


def reporter_node(state: dict) -> dict:
    md = state.get("draft_report") or "(空报告)"
    task_id = state.get("task_id") or ""
    company = state["company"]
    findings = state.get("findings") or []

    # 1) 结构化落库（失败必须显式告警，不能静默吞掉——
    #    否则流程标记 completed 但 DB 无记录，用户无从排查）
    db_ok = True
    try:
        db.save_report(task_id, md)
        db.save_findings(task_id, findings)
    except Exception as e:
        db_ok = False
        logger.warning("reporter 落库失败 task_id=%s: %s", task_id, e)

    # 2) 沉淀进 RAG 知识库（Chroma + FTS5 双写），供后续调研复用
    ingested = 0
    try:
        from knowledge.ingest import ingest_findings, ingest_report
        ingested = ingest_findings(company, task_id, findings)
        ingest_report(company, task_id, md)
    except Exception as e:
        logger.warning("reporter 知识库沉淀失败 task_id=%s: %s", task_id, e)

    if task_id:
        try:
            db.set_task_status(task_id, "completed")
        except Exception as e:
            logger.warning("reporter 状态更新失败 task_id=%s: %s", task_id, e)

    detail = f"入库 {len(findings)} 条情报，知识库沉淀 {ingested} 条"
    if not db_ok:
        detail += "；⚠️ 落库失败，报告未持久化（详见日志）"
    return {
        "final_report": md,
        "timeline": [{
            "node": "reporter", "agent": "reporter", "task": "报告归档",
            "status": "done" if db_ok else "error", "detail": detail,
            "ts": now_str(),
        }],
    }
