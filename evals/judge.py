"""LLM-as-judge：五维结构化评分（与生成模型同档但独立调用，避免自评偏差）。"""
from pydantic import BaseModel, Field

from engine.config import get_llm
from engine.prompts import JUDGE_PROMPT


class JudgeVerdict(BaseModel):
    coverage: int = Field(ge=1, le=5)
    evidence_citation: int = Field(ge=1, le=5)
    structure: int = Field(ge=1, le=5)
    risk_identification: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    overall: float
    rationale: str


def judge_report(task: dict, report: str) -> JudgeVerdict:
    prompt = JUDGE_PROMPT.format(
        company=task["company"],
        objective=task.get("objective", ""),
        key_points="; ".join(task.get("expected_key_points", [])) or "(未指定)",
        report=report or "(空报告)")
    return get_llm("strong").with_structured_output(JudgeVerdict).invoke(prompt)
