"""CLI 入口：命令行跑一次完整调研（便于调试与 CI）。

用法:
    python main.py --company 小米 --mode deep --objective "评估合作机会"
    python main.py --company 小米 --mode standard --auto-approve
"""
import argparse
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph.types import Command  # noqa: E402

from engine.config import settings  # noqa: E402
from engine.events import SINK_NODES, set_event_sink  # noqa: E402
from engine.graph import build_graph  # noqa: E402
from storage import db  # noqa: E402


def _print_ev(ev: dict):
    print(f"  [{ev.get('ts', '')}] {ev.get('node', ''):<16} "
          f"| {ev.get('agent', ''):<14} | {ev.get('task', '')} "
          f"| {ev.get('detail', '')}")


def _cli_sink(task_id: str, ev: dict):
    """实时事件汇：团队子图内部事件即时打印，不等整个子图跑完。"""
    try:
        _print_ev(ev)
    except Exception:
        pass


def _drain(stream) -> bool:
    """消费图的事件流，打印时间线；返回是否因人工审批而暂停。

    经实时事件汇外推的节点（SINK_NODES）的事件已在节点内部打印，
    这里跳过避免重复。
    """
    for chunk in stream:
        for node, update in chunk.items():
            if node == "__interrupt__":
                print("\n[⏸ interrupt] 报告待人工审批，图已暂停（状态已持久化）\n")
                return True
            if isinstance(update, dict):
                if node in SINK_NODES:
                    continue
                for ev in update.get("timeline", []):
                    _print_ev(ev)
    return False


def main():
    parser = argparse.ArgumentParser(description="基于 LangGraph 的多 Agent 企业情报调研系统")
    parser.add_argument("--company", required=True, help="调研目标公司")
    parser.add_argument("--objective", default="", help="调研目标")
    parser.add_argument("--mode", default="standard", choices=["standard", "deep"],
                        help="standard=分层团队模式(B)，deep=深度研究模式(C)")
    parser.add_argument("--auto-approve", action="store_true", help="跳过人工审批（评估/演示用）")
    args = parser.parse_args()

    graph = build_graph(checkpointer=db.get_checkpointer())
    task_id = uuid.uuid4().hex[:12]
    thread_id = uuid.uuid4().hex
    db.create_task(task_id, thread_id, args.company, args.objective, args.mode)
    config = {"configurable": {"thread_id": thread_id},
              "recursion_limit": settings.recursion_limit}
    inputs = {
        "company": args.company,
        "objective": args.objective,
        "mode": args.mode,
        "task_id": task_id,
        "thread_id": thread_id,
        "auto_approve": args.auto_approve,
        "plan": [],
        "findings": [],
        "timeline": [],
        "research_round": 0,
    }

    print(f"▶ 开始调研: {args.company}（模式: {args.mode}）\n")
    set_event_sink(_cli_sink)
    interrupted = _drain(graph.stream(inputs, config, stream_mode="updates"))

    # 多轮审批循环：输入修改意见后 synthesizer 会重新起草并再次 interrupt，
    # 必须循环 resume 直到用户批准（图走到 END）或用户中断，否则图会永远挂起
    while interrupted and not args.auto_approve:
        try:
            ans = input("通过请输入 y，否则输入修改意见: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n(输入中断，图保持暂停状态，thread_id 可用于后续恢复)")
            break
        if not ans:
            continue
        decision = "approve" if ans.lower() in ("y", "yes", "是") else ans
        print()
        interrupted = _drain(graph.stream(Command(resume=decision), config, stream_mode="updates"))

    report = db.get_report(task_id)
    print("\n" + "=" * 60 + "\n最终报告:\n" + "=" * 60 + "\n")
    print(report or "(未生成报告)")


if __name__ == "__main__":
    main()
