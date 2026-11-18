"""麦麦时间上下文插件：两阶段使用同一时间边界，保留原始记录。"""
from collections import OrderedDict
from datetime import datetime
from zoneinfo import ZoneInfo
import asyncio

from maibot_sdk import ConfigWidget, ConfigWidgetState, HookHandler, MaiBotPlugin
from maibot_sdk.types import HookMode, HookOrder

from .config_model import TimeConfig
from .temporal import apply
from .segment_memory import SegmentMemory
from .recall import HistoricalRecall
from .focus import apply_focus


class FeiyuTimePlugin(MaiBotPlugin):
    config_model = TimeConfig
    config_widgets = [ConfigWidget(id="time_status", kind="status", label="当前对话分段"),
        ConfigWidget(id="memory_status", kind="status", label="AI整理进度"),
        ConfigWidget(id="recall_status", kind="status", label="往次聊天召回"),
        ConfigWidget(id="review_memory_now", kind="action", label="让AI检查待整理对话")]

    async def on_load(self):
        self.snapshots = OrderedDict()
        self.last_report = "尚未处理新的模型请求"
        ZoneInfo(self.config.plugin.timezone)
        self.memory = SegmentMemory(self.ctx, self.config.memory, self.ctx.paths.data_dir)
        await self.memory.initialize()
        self.recall = HistoricalRecall(self.memory.client, self.config.memory, self.ctx.logger)
        self.memory_worker = asyncio.create_task(self.memory.run(), name="feiyu-segment-memory")
        self.ctx.logger.info("肥鱼时间上下文已加载：聊天间隔分段、旧摘要降为背景、双向发言时间")

    async def on_unload(self):
        self.memory_worker.cancel()
        await asyncio.gather(self.memory_worker, return_exceptions=True)
        self.snapshots.clear()

    async def on_config_update(self, scope, config_data, version):
        if scope == "self":
            await self.on_unload()
            await self.on_load()
            return
        ZoneInfo(self.config.plugin.timezone)
        self.snapshots.clear()

    def process(self, kwargs, replyer=False):
        cfg = self.config.plugin
        if not cfg.enabled:
            return {"action": "continue"}
        if kwargs.get("item_schema_version") != 1 or not isinstance(kwargs.get("items"), list):
            raise ValueError("时间上下文要求 Context Item schema 1")
        session = str(kwargs.get("session_id", ""))
        if not session:
            return {"action": "continue"}
        items, snapshot, report = apply(kwargs["items"], datetime.now(ZoneInfo(cfg.timezone)),
            idle_minutes=cfg.idle_minutes, recent_minutes=cfg.recent_minutes,
            background_chars=cfg.background_chars,
            remembered=self.snapshots.get(session) if replyer else None,
            reply_message_id=str(kwargs.get("reply_message_id", "")) if replyer else "")
        if not replyer:
            self.snapshots[session] = snapshot
            self.snapshots.move_to_end(session)
            while len(self.snapshots) > 64:
                self.snapshots.popitem(last=False)
        items = self.recall.attach(session, items, datetime.now(ZoneInfo(cfg.timezone)),
                                  str(kwargs.get('reply_message_id', '')) if replyer else '')
        items = apply_focus(items, datetime.now(ZoneInfo(cfg.timezone)),
                            remembered=self.snapshots.get(session))
        self.last_report = f"{'回复器' if replyer else '规划器'}：{report['input_items']} → {report['output_items']} 条；当前边界 {report['boundary']}"
        self.ctx.logger.info("时间上下文 %s session=%s", self.last_report, session)
        return {"action": "continue", "modified_kwargs": {**kwargs, "items": items}}

    @HookHandler("maisaka.planner.before_request", name="temporal_planner", mode=HookMode.BLOCKING,
                 order=HookOrder.LATE, timeout_ms=5000)
    async def planner(self, **kwargs):
        if self.config.plugin.enabled:
            await self.recall.prepare(str(kwargs.get('session_id', '')), kwargs.get('items', []),
                                      datetime.now(ZoneInfo(self.config.plugin.timezone)))
        return self.process(kwargs)

    @HookHandler("maisaka.replyer.before_model_request", name="temporal_replyer", mode=HookMode.BLOCKING,
                 order=HookOrder.LATE, timeout_ms=2000)
    async def replyer(self, **kwargs):
        return self.process(kwargs, replyer=True)

    async def get_config_widget_state(self):
        return {"time_status": ConfigWidgetState(text=self.last_report, tone="neutral"),
                "memory_status": ConfigWidgetState(text=self.memory.status, tone="neutral"),
                "recall_status": ConfigWidgetState(text=self.recall.status, tone="neutral"),
                "review_memory_now": ConfigWidgetState()}

    async def on_config_widget_action(self, action_id):
        if action_id != "review_memory_now":
            raise ValueError("未知操作")
        self.memory.force = True
        self.memory.wakeup.set()


def create_plugin():
    return FeiyuTimePlugin()
