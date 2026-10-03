# 企业情报调研分析系统

基于 **LangGraph** 的多 Agent 企业情报调研系统：给定一家公司，自动完成情报搜集 → 分析 → 风险评估 → 综合报告，全程支持人工审批（human-in-the-loop）与断点续跑，调研产出自动沉淀进可复用的 RAG 情报知识库。

## 功能特性

- **双模式编排**
  - `standard`（分层团队）：搜集组（搜索/工商/专利）→ 分析组 ∥ 风控组（并行）→ 综合报告
  - `deep`（深度研究）：规划器拆解子问题 → 并行研究员 ×N → Critic 反思质检循环（按缺口补充子问题，最多 `MAX_RESEARCH_ROUNDS` 轮）
- **人工审批门控**：报告草稿生成后通过 LangGraph `interrupt` 暂停，支持多轮「修改意见 → 重新起草」，图状态持久化，刷新/重启不丢失
- **混合检索知识库**：Chroma 向量召回 + SQLite FTS5(BM25) 双路召回，RRF 融合排序；调研发现与历史报告自动双写沉淀（内容摘要幂等，重跑不重复）
- **LLM 分档路由**：strong（规划/质检/综合）与 cheap（组长拆解/摘要）双档，兼容任意 OpenAI 兼容服务（DeepSeek / 硅基流动 / OneAPI / vLLM）
- **降级设计**：搜索无 Key → 模拟数据；Embedding 未配置 → 纯 BM25 兜底；起草失败 → 确定性降级稿；质检异常 → 放行防死循环
- **Vue 3 前端**：实时 Agent 时间线（2s 轮询）、语义色状态系统、审批面板、报告渲染与下载、知识库检索

## 架构

```
                        ┌─────────────────────────────────────┐
                        │           Vue 3 SPA (web/)          │
                        │  时间线轮询 · 审批 · 报告 · 知识库检索  │
                        └──────────────┬──────────────────────┘
                                       │ REST (JSON)
                        ┌──────────────▼──────────────────────┐
                        │        FastAPI (api.py)             │
                        │  后台线程跑图 · 状态机 · 事件落库      │
                        └──────────────┬──────────────────────┘
                                       │
        ┌──────────────────────────────▼──────────────────────────────┐
        │                  LangGraph 主图 (engine/graph.py)            │
        │                                                             │
        │  START → intake ─┬─ standard ─→ collection_team             │
        │                  │                ├→ analysis_team ┐        │
        │                  │                └→ risk_team ─────┤        │
        │                  └─ deep ─→ deep_planner → researcher×N      │
        │                             ↑              │                 │
        │                             └── critic ◄───┘                 │
        │                                                             │
        │      → synthesizer → human_review(interrupt) → reporter     │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
             ┌─────────────┬───────────┴──────────┬─────────────┐
             │  SQLite     │   Chroma (向量)       │  Tavily /   │
             │  任务/事件/   │   + FTS5 (BM25)      │  网页抓取    │
             │  报告/缓存   │   RRF 混合检索        │             │
             └─────────────┴──────────────────────┴─────────────┘
```

## 目录结构

```
├── api.py               # FastAPI 后端：REST API + 静态托管 + 后台执行
├── main.py              # CLI 入口（调试/CI 用）
├── engine/              # 编排核心
│   ├── graph.py         #   主图组装（双模式）
│   ├── state.py         #   共享状态（findings/timeline 带 add_list reducer）
│   ├── config.py        #   配置与 LLM 双档路由
│   ├── nodes/           #   intake / deep_planner / critic / synthesizer /
│   │                    #   human_review(interrupt) / reporter
│   ├── teams/           #   团队子图工厂（组长拆解 → Send 并行 → 质检修订）
│   ├── agents/          #   ReAct worker 构建器
│   └── prompts.py       #   全部提示词
├── knowledge/           # 向量库封装 / 情报入库 / RRF 混合检索
├── storage/             # SQLite DAO + schema + checkpointer 工厂
├── tools/               # tavily_search（带缓存）/ web_crawl / 工商·专利模拟
├── web/                 # Vue 3 前端（免构建，vendor 已本地化）
│   ├── index.html / css/app.css / js/app.js
│   └── vendor/          # vue / marked / DOMPurify（本地文件，离线可用）
├── evals/               # LLM-as-judge 评估流水线（五维评分）
└── tests/               # 单测 + FakeModel 全图 e2e（零 API 成本）
```

## 快速开始

```bash
# 1. 安装依赖（Python 3.10+）
pip install -r requirements.txt

# 2. 配置 .env（参考 .env.example）
cp .env.example .env
# 至少填: OPENAI_API_KEY（或 DEEPSEEK_API_KEY）

# 3. 启动
uvicorn api:app --port 8000
# 浏览器打开 http://localhost:8000
```

CLI 方式（调试/CI）：

```bash
python main.py --company 小米 --mode standard --auto-approve
python main.py --company 小米 --mode deep --objective "评估合作机会"
```

测试与评估（零 API 成本）：

```bash
python -m pytest tests/ -q        # 15 个用例：门控/RRF/DAO/双模式全图 e2e
python -m evals.run_eval --limit 3 # LLM-as-judge 五维评估（消耗 API）
```

## 配置项（.env）

| 变量 | 必填 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` | ✅ | LLM Key，任意 OpenAI 兼容服务 |
| `OPENAI_BASE_URL` / `DEEPSEEK_BASE_URL` | | 网关地址，默认 DeepSeek 官方 |
| `STRONG_MODEL` / `CHEAP_MODEL` | | 模型分档，默认 `deepseek-chat` |
| `TAVILY_API_KEY` | | 联网搜索；不填则用确定性模拟数据（可离线演示） |
| `EMBEDDING_API_KEY` | | OpenAI 兼容 embedding API；不填则纯 BM25 兜底 |
| `EMBEDDING_BASE_URL` | | 默认硅基流动 `https://api.siliconflow.cn/v1` |
| `EMBEDDING_MODEL` | | 默认 `BAAI/bge-m3`（免费、中文效果好） |
| `MAX_RESEARCH_ROUNDS` | | deep 模式反思循环上限，默认 3 |
| `RECURSION_LIMIT` | | LangGraph 递归上限，默认 80 |

## REST API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/tasks` | 发起调研（后台线程执行），返回 task_id |
| GET | `/api/tasks` | 历史任务列表 |
| GET | `/api/tasks/{id}` | 任务详情（含状态） |
| GET | `/api/tasks/{id}/events` | Agent 时间线（前端轮询源） |
| GET | `/api/tasks/{id}/result` | 最终报告 + 情报明细 |
| GET | `/api/tasks/{id}/approval` | interrupt 审批快照（草稿 + 风险信号） |
| POST | `/api/tasks/{id}/resume` | 提交审批决策（`approve` 或修改意见文本） |
| GET | `/api/kb/search?q=` | 知识库混合检索 |

任务状态机：`running → awaiting_approval | completed | failed`

## 设计要点

- **并行写安全**：`findings` / `timeline` 声明 `add_list` reducer，Send fan-out 不会写冲突（`engine/state.py`）
- **子图状态隔离**：`wrap_team` 只回传 findings/timeline，团队内部字段不污染父图（`engine/teams/common.py`）
- **审批可恢复**：interrupt 依赖 `thread_id` 稳定，checkpointer 落 `data/checkpoints.sqlite`，服务重启后仍可审批续跑
- **双写幂等**：chunk_id = 内容摘要 md5，向量侧 upsert、FTS 侧先删后插，重复调研不产生重复情报
- **前端语义色**：蓝=搜集 / 紫=分析 / 琥珀=风险·待审批 / 青=深度 / 红=失败 / 绿=完成，颜色承载信息而非装饰
