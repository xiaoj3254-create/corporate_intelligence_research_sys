"""实时事件汇：长耗时节点内部事件的即时外推通道。

背景: 团队子图（拆解 → Send 并行 workers → 质检 → 可能的修订轮）与 deep 模式
的并行 researcher 都可能在单个"原子节点"内部执行数分钟。宿主若只消费主图
stream，要等整个节点（Send 场景下甚至是本 superstep 里最慢的并行实例）结束
才有输出；注册 sink 后事件在节点内部产生的瞬间即被外推——api 用它落库供
前端轮询，CLI 用它即时打印。

去重契约: 经 sink 外推过的事件，随后仍会随节点 update 进入主图 stream
（状态里的 timeline 需保留完整历史）。消费端 _drain 必须跳过 SINK_NODES
中节点的 timeline，避免重复落库/打印。
"""
import logging

logger = logging.getLogger(__name__)

# 事件经 sink 实时外推的节点集合；主图 stream 消费端据此去重
SINK_NODES = frozenset({"collection_team", "analysis_team", "risk_team",
                        "researcher"})

_event_sink = None


def set_event_sink(cb):
    """注册实时事件回调 cb(task_id, ev)；传 None 取消。"""
    global _event_sink
    _event_sink = cb


def emit_live(task_id, ev):
    """把事件即时推给宿主。

    未注册 sink / task_id 为空 / 回调异常时静默跳过——实时外推是锦上添花，
    任何情况下都不允许连累图执行。
    """
    if _event_sink and task_id:
        try:
            _event_sink(task_id, ev)
        except Exception as e:
            logger.debug("emit_live 回调异常 task_id=%s: %s", task_id, e)
