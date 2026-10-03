"""SQLite 连接管理 + DAO + LangGraph checkpointer 工厂。

并发说明: FastAPI 后台线程 + API 请求线程共享进程级单连接。
所有读写都经过模块级 RLock 串行化，避免并发写触发
'database is locked' 或交叉 commit。
"""
import json
import logging
import os
import sqlite3
import threading

from engine.config import settings

logger = logging.getLogger(__name__)

_conn = None
_lock = threading.RLock()
_cp_conn = None
_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")


def get_conn() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            os.makedirs(settings.data_dir, exist_ok=True)
            _conn = sqlite3.connect(
                os.path.join(settings.data_dir, "intel.db"), check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            with open(_SCHEMA_PATH, encoding="utf-8") as f:
                _conn.executescript(f.read())
            _conn.commit()
        return _conn


def _write(sql: str, params: tuple = ()):
    """串行化的单语句写操作。"""
    with _lock:
        conn = get_conn()
        conn.execute(sql, params)
        conn.commit()


def get_checkpointer():
    """LangGraph SqliteSaver：状态持久化 / 断点续跑 / interrupt 恢复。

    连接做模块级单例：此前每次调用都新建裸连接且从不关闭，长进程会泄漏句柄。
    """
    global _cp_conn
    from langgraph.checkpoint.sqlite import SqliteSaver

    with _lock:
        if _cp_conn is None:
            os.makedirs(settings.data_dir, exist_ok=True)
            _cp_conn = sqlite3.connect(
                os.path.join(settings.data_dir, "checkpoints.sqlite"),
                check_same_thread=False)
        return SqliteSaver(_cp_conn)


# ---------------------------------------------------------------- tasks

def create_task(task_id: str, thread_id: str, company: str, objective: str, mode: str):
    _write("INSERT INTO tasks (id, thread_id, company, objective, mode) VALUES (?,?,?,?,?)",
           (task_id, thread_id, company, objective, mode))


def set_task_status(task_id: str, status: str):
    _write("UPDATE tasks SET status=? WHERE id=?", (status, task_id))


def get_task(task_id: str):
    with _lock:
        row = get_conn().execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return dict(row) if row else None


def list_tasks(limit: int = 30):
    with _lock:
        rows = get_conn().execute(
            "SELECT * FROM tasks ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------- events

def save_event(task_id: str, ev: dict):
    _write(
        "INSERT INTO task_events (task_id, node, agent, task_desc, status, detail, ts) "
        "VALUES (?,?,?,?,?,?,?)",
        (task_id, ev.get("node", ""), ev.get("agent", ""), ev.get("task", ""),
         ev.get("status", "done"), ev.get("detail", ""), ev.get("ts", "")))


def list_events(task_id: str):
    with _lock:
        rows = get_conn().execute(
            "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (task_id,)).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------- findings

def save_findings(task_id: str, findings):
    with _lock:
        conn = get_conn()
        for f in findings:
            conn.execute(
                "INSERT INTO findings (task_id, team, agent, task_desc, content, source_urls, credibility) "
                "VALUES (?,?,?,?,?,?,?)",
                (task_id, f.team, f.agent, f.task, f.content,
                 json.dumps(f.source_urls, ensure_ascii=False), f.credibility))
        conn.commit()


def list_findings(task_id: str):
    with _lock:
        rows = get_conn().execute(
            "SELECT * FROM findings WHERE task_id=? ORDER BY id", (task_id,)).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------- reports

def save_report(task_id: str, markdown: str):
    _write("INSERT OR REPLACE INTO reports (task_id, markdown) VALUES (?,?)",
           (task_id, markdown))


def get_report(task_id: str):
    with _lock:
        row = get_conn().execute(
            "SELECT markdown FROM reports WHERE task_id=?", (task_id,)).fetchone()
        return row["markdown"] if row else None


# ---------------------------------------------------------------- search cache

# 缓存有效期：搜索结果本身有时效性，过期后重新检索
CACHE_TTL_DAYS = 7


def cache_get(key: str):
    with _lock:
        row = get_conn().execute(
            "SELECT result_text, created_at FROM search_cache WHERE query_hash=?",
            (key,)).fetchone()
        if not row:
            return None
        try:
            from datetime import datetime as _dt
            created = _dt.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S")
            if (_dt.now() - created).days >= CACHE_TTL_DAYS:
                return None
        except (ValueError, TypeError):
            return None
        return row["result_text"]


def cache_put(key: str, text: str):
    _write("INSERT OR REPLACE INTO search_cache (query_hash, result_text) VALUES (?,?)",
           (key, text))


# ---------------------------------------------------------------- FTS5 (BM25)

def fts_insert(content: str, company: str, chunk_id: str):
    """同 chunk_id 先删后插（幂等），与 knowledge/ingest.py 的双写路径保持一致。

    chunks_fts 是无唯一约束的虚拟表，若用纯 INSERT，重跑会产生重复 chunk。
    """
    with _lock:
        conn = get_conn()
        conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
        conn.execute(
            "INSERT INTO chunks_fts (content, company, chunk_id) VALUES (?,?,?)",
            (content, company or "", chunk_id))
        conn.commit()


def fts_search(query: str, company: str = None, limit: int = 20):
    """BM25 检索，返回 [{chunk_id, content}, ...]，按相关度升序（越小越相关）。"""
    with _lock:
        # 用双引号包裹做短语匹配，避免用户输入触发 FTS 语法错误
        match = '"' + query.replace('"', '""') + '"'
        sql = "SELECT chunk_id, content FROM chunks_fts WHERE chunks_fts MATCH ?"
        args = [match]
        if company:
            sql += " AND company = ?"
            args.append(company)
        sql += " ORDER BY bm25(chunks_fts) LIMIT ?"
        args.append(limit)
        return [dict(r) for r in get_conn().execute(sql, args).fetchall()]


# ---------------------------------------------------------------- evals

def save_eval(task_id: str, company: str, judge_model: str, verdict):
    with _lock:
        conn = get_conn()
        scores = {
            "coverage": verdict.coverage,
            "evidence_citation": verdict.evidence_citation,
            "structure": verdict.structure,
            "risk_identification": verdict.risk_identification,
            "actionability": verdict.actionability,
        }
        conn.execute(
            "INSERT INTO eval_results (task_id, company, judge_model, dimension_scores, overall, rationale) "
            "VALUES (?,?,?,?,?,?)",
            (task_id, company, judge_model, json.dumps(scores), verdict.overall, verdict.rationale))
        conn.commit()


def list_evals(limit: int = 50):
    with _lock:
        rows = get_conn().execute(
            "SELECT * FROM eval_results ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
