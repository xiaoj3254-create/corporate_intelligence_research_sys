"""FastAPI 后端：为 Vue 前端提供 REST API。

启动:
    uvicorn api:app --port 8000
    然后浏览器打开 http://localhost:8000

职责:
- 发起调研（后台线程执行 graph.stream，事件实时落库）
- 时间线/报告/知识库查询（前端轮询）
- interrupt 审批快照 + resume

并发说明: SqliteSaver 的 SQLite 连接不保证跨线程并发安全，
演示/单机场景用全局锁串行化"图执行"，查询接口不受影响。
"""
import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine.config import settings
from engine.graph import build_graph
from langgraph.types import Command
from storage import db

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

WEB_DIR = Path(__file__).parent / "web"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """startup 时打印 embedding 配置状态（替代已弃用的 @app.on_event）。"""
    if settings.embedding_api_key:
        logger.info("embedding: %s @ %s（向量召回已启用）",
                    settings.embedding_model, settings.embedding_base_url)
    else:
        logger.warning("未配置 EMBEDDING_API_KEY，知识库将使用纯 BM25 检索")
    yield


app = FastAPI(title="企业情报调研系统 API", lifespan=lifespan)

# 图执行全局锁：同一时刻只跑一次调研（checkpointer 连接非线程安全）
_RUN_LOCK = threading.Lock()
_graph = None

# kb_search 专用常驻线程池。不能用 with ThreadPoolExecutor(...)：
# 其 __exit__ 的 shutdown(wait=True) 会阻塞到检索线程真正结束，
# 使超时兜底失效（最坏接口挂 ~60s）。改为常驻池后，超时立即返回 503，
# 孤儿线程由 embedding 请求自身的 timeout=30 兜底结束；
# max_workers=2 允许一个卡死的检索期间新请求仍可入队执行。
_KB_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kb-search")


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(checkpointer=db.get_checkpointer())
    return _graph


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id},
            "recursion_limit": settings.recursion_limit}


def _save_event(task_id: str, node: str, agent: str, task: str,
                status: str, detail: str):
    try:
        db.save_event(task_id, {"node": node, "agent": agent, "task": task,
                                "status": status, "detail": detail, "ts": _now()})
    except Exception as e:
        logger.warning("save_event 失败 task_id=%s: %s", task_id, e)


def _drain(task_id: str, thread_id: str, stream) -> bool:
    """消费事件流：逐条落库（前端轮询即可"实时"）。返回是否 interrupt 暂停。"""
    interrupted = False
    for chunk in stream:
        for node, update in chunk.items():
            if node == "__interrupt__":
                interrupted = True
                _save_event(task_id, "human_review", "人工审批", "报告待审批",
                            "pending", "系统已暂停，等待人工审批（interrupt）")
                continue
            if not isinstance(update, dict) or not update:
                continue
            for ev in update.get("timeline") or []:
                ev.setdefault("ts", _now())
                ev.setdefault("status", "done")
                try:
                    db.save_event(task_id, ev)
                except Exception as e:
                    logger.warning("save_event 失败: %s", e)
    return interrupted


def _run_background(task_id: str, thread_id: str, company: str, objective: str,
                    mode: str, auto_approve: bool, resume_decision: str = None):
    """后台线程：跑（或恢复跑）一次调研，维护任务状态机。

    状态: running → awaiting_approval | completed | failed
    """
    try:
        with _RUN_LOCK:
            graph = _get_graph()
            if resume_decision is None:
                inputs = {
                    "company": company, "objective": objective, "mode": mode,
                    "task_id": task_id, "thread_id": thread_id,
                    "auto_approve": auto_approve,
                    "plan": [], "findings": [], "timeline": [], "research_round": 0,
                }
                stream = graph.stream(inputs, _config(thread_id), stream_mode="updates")
            else:
                stream = graph.stream(Command(resume=resume_decision),
                                      _config(thread_id), stream_mode="updates")
            interrupted = _drain(task_id, thread_id, stream)

        if interrupted and not auto_approve:
            db.set_task_status(task_id, "awaiting_approval")
        elif db.get_task(task_id) and db.get_task(task_id)["status"] != "completed":
            db.set_task_status(task_id, "completed")
    except Exception as e:
        logger.exception("调研执行失败 task_id=%s", task_id)
        _save_event(task_id, "system", "system", "执行异常", "error", str(e)[:500])
        try:
            db.set_task_status(task_id, "failed")
        except Exception:
            pass


# ---------------------------------------------------------------- schemas

class TaskCreate(BaseModel):
    company: str = Field(min_length=1, max_length=100)
    objective: str = Field(default="", max_length=2000)
    mode: str = Field(default="standard", pattern="^(standard|deep)$")
    auto_approve: bool = False


class Decision(BaseModel):
    decision: str = Field(min_length=1, max_length=2000)


# ---------------------------------------------------------------- API

@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/tasks")
def create_task(body: TaskCreate):
    task_id = uuid.uuid4().hex[:12]
    thread_id = uuid.uuid4().hex
    db.create_task(task_id, thread_id, body.company.strip(),
                   body.objective.strip(), body.mode)
    t = threading.Thread(
        target=_run_background,
        args=(task_id, thread_id, body.company.strip(), body.objective.strip(),
              body.mode, body.auto_approve),
        daemon=True, name=f"research-{task_id}")
    t.start()
    return {"task_id": task_id, "thread_id": thread_id}


@app.get("/api/tasks")
def list_tasks():
    return db.list_tasks(50)


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    return task


@app.get("/api/tasks/{task_id}/events")
def get_events(task_id: str):
    return db.list_events(task_id)


@app.get("/api/tasks/{task_id}/result")
def get_result(task_id: str):
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    findings = db.list_findings(task_id)
    for f in findings:
        try:
            f["source_urls"] = json.loads(f.get("source_urls") or "[]")
        except Exception:
            f["source_urls"] = []
    return {"report": db.get_report(task_id), "findings": findings}


@app.get("/api/tasks/{task_id}/approval")
def get_approval(task_id: str):
    """interrupt 暂停时的审批快照：是否等待中 + 草稿 + 全部 findings。"""
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    try:
        graph = _get_graph()
        # get_state 与后台 stream 共享同一个 SqliteSaver 连接（非线程安全）。
        # 用带超时的 _RUN_LOCK 竞争：任务执行中拿不到锁（~2s）直接按"未在
        # 审批点"返回——此时本就不存在可审批的 interrupt；awaiting_approval
        # 期间无 stream 在跑，锁必然可立即获取，读到真实快照。
        if not _RUN_LOCK.acquire(timeout=2):
            return {"awaiting": False, "draft": "", "findings": []}
        try:
            snap = graph.get_state(_config(task["thread_id"]))
        finally:
            _RUN_LOCK.release()
        awaiting = "human_review" in (snap.next or ())
        vals = snap.values or {}
        return {"awaiting": awaiting,
                "draft": vals.get("draft_report") or "",
                "findings": [f.model_dump() for f in (vals.get("findings") or [])]}
    except Exception as e:
        logger.warning("读取图状态失败 task_id=%s: %s", task_id, e)
        return {"awaiting": False, "draft": "", "findings": []}


@app.post("/api/tasks/{task_id}/resume")
def resume_task(task_id: str, body: Decision):
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    # 只有真正停在 interrupt 点的任务才能 resume：否则 Command(resume=...)
    # 会在图上抛异常，后台线程把任务误标为 failed
    if task["status"] != "awaiting_approval":
        raise HTTPException(
            409, f"任务当前状态为 {task['status']}，仅 awaiting_approval 状态可提交审批")
    db.set_task_status(task_id, "running")
    t = threading.Thread(
        target=_run_background,
        args=(task_id, task["thread_id"], task["company"], task["objective"],
              task["mode"], False, body.decision.strip()),
        daemon=True, name=f"resume-{task_id}")
    t.start()
    return {"ok": True}


@app.get("/api/kb/search")
def kb_search(q: str, k: int = 8):
    from knowledge.hybrid_retriever import hybrid_retrieve

    def _search():
        return hybrid_retrieve(q.strip(), k=max(1, min(k, 20)))

    # 向量召回依赖第三方 embedding API（未配置时内部自动降级为纯 BM25）；
    # 提交到常驻线程池，30s 内未返回即放弃（孤儿线程自行结束后释放），
    # 防止外部 API 卡死拖垮接口
    future = _KB_POOL.submit(_search)
    try:
        results = future.result(timeout=30)
    except FutTimeout:
        future.cancel()
        raise HTTPException(503, "检索超时（embedding 服务未响应），请稍后重试")
    except Exception as e:
        raise HTTPException(500, f"检索失败: {e}")
    return {"results": results}


# ---------------------------------------------------------------- 静态前端

@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
