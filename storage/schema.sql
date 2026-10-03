-- 任务表（thread_id 关联 LangGraph checkpointer，支持断点续跑）
CREATE TABLE IF NOT EXISTS tasks (
    id         TEXT PRIMARY KEY,
    thread_id  TEXT UNIQUE,
    company    TEXT NOT NULL,
    objective  TEXT,
    mode       TEXT CHECK (mode IN ('standard', 'deep')),
    status     TEXT DEFAULT 'running',
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

-- Agent 执行时间线事件（前端时间线视图数据源）
CREATE TABLE IF NOT EXISTS task_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id   TEXT,
    node      TEXT,
    agent     TEXT,
    task_desc TEXT,
    status    TEXT,
    detail    TEXT,
    ts        TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_task ON task_events(task_id);

-- 调研发现明细
CREATE TABLE IF NOT EXISTS findings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT,
    team        TEXT,
    agent       TEXT,
    task_desc   TEXT,
    content     TEXT,
    source_urls TEXT,
    credibility TEXT,
    created_at  TEXT DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS idx_findings_task ON findings(task_id);

-- 最终报告
CREATE TABLE IF NOT EXISTS reports (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT UNIQUE,
    markdown   TEXT,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

-- Tavily 搜索缓存（避免演示/评估时重复消耗配额）
CREATE TABLE IF NOT EXISTS search_cache (
    query_hash  TEXT PRIMARY KEY,
    result_text TEXT,
    created_at  TEXT DEFAULT (datetime('now', 'localtime'))
);

-- LLM-as-judge 评估结果
CREATE TABLE IF NOT EXISTS eval_results (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id          TEXT,
    company          TEXT,
    judge_model      TEXT,
    dimension_scores TEXT,
    overall          REAL,
    rationale        TEXT,
    created_at       TEXT DEFAULT (datetime('now', 'localtime'))
);

-- BM25 全文索引（与 Chroma 向量检索做 RRF 混合）
-- trigram 分词器: 支持中文子串匹配（unicode61 会把整句中文当作单 token）
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content, company, chunk_id, tokenize='trigram');
