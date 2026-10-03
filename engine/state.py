"""全部共享状态与数据模型定义。

关键点: 所有被并行节点写入的字段（findings / timeline）
必须声明 reducer（add_list），否则并行 Send 会触发写冲突。
"""
from datetime import datetime
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


def add_list(left, right):
    """并行写入安全的列表合并 reducer（容忍 None / 缺省）。"""
    return list(left or []) + list(right or [])


def now_str() -> str:
    """时间线事件时间戳。带完整日期，跨天/跨年排查不歧义。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class SubQuestion(BaseModel):
    """深度模式的子问题。"""
    id: str = ""
    question: str
    priority: int = Field(default=3, ge=1, le=5)


class Finding(BaseModel):
    """调研发现：所有 Agent 的统一产出单元。"""
    team: Literal["collection", "analysis", "risk", "deep"]
    agent: str
    task: str = ""
    content: str
    source_urls: list[str] = Field(default_factory=list)
    credibility: Literal["high", "medium", "low"] = "medium"


class Task(BaseModel):
    """团队子图内的执行任务。"""
    id: str = ""
    type: str
    description: str


class TaskList(BaseModel):
    tasks: list[Task] = Field(default_factory=list)


class ReviewResult(BaseModel):
    pass_ok: bool = True
    missing_tasks: list[Task] = Field(default_factory=list)


class PlanOutput(BaseModel):
    sub_questions: list[SubQuestion] = Field(default_factory=list)


class QualityAssessment(BaseModel):
    """Critic 反思质检的结构化结论。"""
    overall_pass: bool = False
    coverage_gaps: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    dimension_scores: dict[str, int] = Field(default_factory=dict)


class MainState(TypedDict):
    """主图状态。"""
    company: str
    objective: str
    mode: str                                   # standard | deep
    task_id: str
    thread_id: str
    auto_approve: bool

    plan: list[SubQuestion]                     # 深度模式子问题
    findings: Annotated[list[Finding], add_list]
    research_round: int
    critic_verdict: Optional[QualityAssessment]

    draft_report: Optional[str]
    human_feedback: Optional[str]               # "approve" | 修改意见
    final_report: Optional[str]

    timeline: Annotated[list[dict], add_list]


class TeamState(TypedDict):
    """团队子图状态（findings/timeline 与主图同名同 reducer，自动映射）。"""
    company: str
    objective: str
    prior_findings: list[Finding]               # 上游团队（搜集组）的产出
    tasks: list[Task]
    task: Task                                  # Send 注入: 单个 worker 的任务
    findings: Annotated[list[Finding], add_list]
    timeline: Annotated[list[dict], add_list]
    quality_ok: bool
    revise_count: int
