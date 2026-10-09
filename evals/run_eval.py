"""评估流水线 CLI：跑评测集 → LLM-as-judge 打分 → 落库 + 汇总输出。

用法:
    python -m evals.run_eval
    python -m evals.run_eval --limit 3
"""
import argparse
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.config import settings  # noqa: E402
from engine.graph import build_graph  # noqa: E402
from evals.judge import judge_report  # noqa: E402
from storage import db  # noqa: E402

DIMS = ["coverage", "evidence_citation", "structure",
        "risk_identification", "actionability"]


def run_research(graph, spec: dict):
    """跑完整调研（auto_approve 跳过人工审批），返回 (报告, task_id)。"""
    task_id = uuid.uuid4().hex[:12]
    thread_id = uuid.uuid4().hex
    # kind='eval': 评估任务与正式任务同表存储（报告/发现照常落库可追溯），
    # 但 list_tasks 会过滤，不污染前端任务历史
    db.create_task(task_id, thread_id, spec["company"],
                   spec.get("objective", ""), spec.get("mode", "deep"),
                   kind="eval")
    inputs = {
        "company": spec["company"],
        "objective": spec.get("objective", ""),
        "mode": spec.get("mode", "deep"),
        "task_id": task_id,
        "thread_id": thread_id,
        "auto_approve": True,   # 评估场景自动通过审批，无需 checkpointer
        "plan": [],
        "findings": [],
        "timeline": [],
        "research_round": 0,
    }
    config = {"configurable": {"thread_id": thread_id},
              "recursion_limit": settings.recursion_limit}
    final = {}
    for chunk in graph.stream(inputs, config, stream_mode="values"):
        final = chunk
    report = final.get("final_report") or final.get("draft_report") or ""
    return report, task_id


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge 评估流水线")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个任务（0=全部）")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "dataset", "eval_tasks.json"), encoding="utf-8") as f:
        specs = json.load(f)
    if args.limit:
        specs = specs[:args.limit]

    graph = build_graph()  # 无 checkpointer（auto_approve 下不会 interrupt）
    results = []
    for spec in specs:
        print(f"▶ 调研: {spec['company']}（{spec.get('mode', 'deep')}）…")
        try:
            report, task_id = run_research(graph, spec)
            verdict = judge_report(spec, report)
            db.save_eval(task_id, spec["company"], settings.strong_model, verdict)
            results.append((spec["company"], verdict))
            print(f"   五维: {[getattr(verdict, d) for d in DIMS]}  "
                  f"overall={verdict.overall}")
        except Exception as e:
            print(f"   ✗ 失败: {e}")

    if not results:
        print("无评估结果")
        return

    print("\n" + "=" * 78)
    print(f"{'公司':<10}{'coverage':>9}{'evidence':>9}{'structure':>10}"
          f"{'risk':>7}{'action':>8}{'overall':>9}")
    print("-" * 78)
    sums = {d: 0 for d in DIMS}
    for company, v in results:
        print(f"{company:<10}{v.coverage:>9}{v.evidence_citation:>9}{v.structure:>10}"
              f"{v.risk_identification:>7}{v.actionability:>8}{v.overall:>9}")
        for d in DIMS:
            sums[d] += getattr(v, d)
    print("-" * 78)
    n = len(results)
    avg = "  ".join(f"{d}={sums[d] / n:.2f}" for d in DIMS)
    print(f"均分({n} 个任务): {avg}  overall="
          f"{sum(v.overall for _, v in results) / n:.2f}")


if __name__ == "__main__":
    main()
