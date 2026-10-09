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


def _migrate(conn: sqlite3.Connection):
    """轻量列迁移：schema.sql 只对新建库生效，旧库缺列时在此补齐（已存在则忽略）。"""
    try:
        conn.execute("ALTER TABLE tasks ADD COLUMN kind TEXT NOT NULL DEFAULT 'task'")
    except sqlite3.OperationalError:
        pass


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
            _migrate(_conn)
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


def new_checkpointer():
    """新建独立连接的 SqliteSaver（WAL 模式）。

    每个后台调研任务专用一个连接，任务结束由调用方关闭：
    此前全局单连接非线程安全，只能靠 _RUN_LOCK 把图执行串行化，
    第二个任务会静默排队数分钟。WAL 允许多连接并发读写。
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    os.makedirs(settings.data_dir, exist_ok=True)
    conn = sqlite3.connect(
        os.path.join(settings.data_dir, "checkpoints.sqlite"),
        check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    return SqliteSaver(conn)


# ---------------------------------------------------------------- tasks

def create_task(task_id: str, thread_id: str, company: str, objective: str,
                mode: str, kind: str = "task"):
    """kind: 'task'=正式任务（列表可见）| 'eval'=评估任务（列表过滤）。"""
    _write("INSERT INTO tasks (id, thread_id, company, objective, mode, kind) "
           "VALUES (?,?,?,?,?,?)",
           (task_id, thread_id, company, objective, mode, kind))


def set_task_status(task_id: str, status: str):
    _write("UPDATE tasks SET status=? WHERE id=?", (status, task_id))


def get_task(task_id: str):
    with _lock:
        row = get_conn().execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return dict(row) if row else None


def list_tasks(limit: int = 30):
    """任务历史（只含正式任务；eval 评估任务不入列表，避免污染前端）。"""
    with _lock:
        rows = get_conn().execute(
            "SELECT * FROM tasks WHERE kind='task' "
            "ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def delete_task(task_id: str):
    """删除任务及其全部关联数据（时间线/发现/报告）。"""
    with _lock:
        conn = get_conn()
        for table in ("task_events", "findings", "reports"):
            conn.execute(f"DELETE FROM {table} WHERE task_id=?", (task_id,))
        conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        conn.commit()


def fail_stale_running_tasks() -> int:
    """服务启动时清理僵尸任务：进程重启后 DB 中残留的 running 状态任务
    必然没有线程在执行（后台线程随进程消亡），统一标记为 failed。
    返回清理数量。"""
    with _lock:
        conn = get_conn()
        cur = conn.execute("UPDATE tasks SET status='failed' WHERE status='running'")
        conn.commit()
        return cur.rowcount


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
