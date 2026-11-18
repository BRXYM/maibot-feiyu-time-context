"""仅整理本轮请求副本；不写聊天数据库，不改变发送对象或工具参数。"""
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import unescape
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo
import json
import re

MESSAGE = re.compile(r'^<message\s+([^>]+)>\s*\n?', re.S)
ATTR = re.compile(r'(\w+)="([^"]*)"')
SUMMARY = "【上下文状态摘要-内部参考】"
CLOCK = "【时间与当前对话】"
BACKGROUND = "【往次对话资料"
SINGING = "【本私聊学唱任务实时状态"
TOOL_TYPES = {"FunctionCallItem", "FunctionCallOutputItem"}


def text(item: dict) -> str:
    return "\n".join(p.get("text", "") for p in item.get("parts", []) if p.get("type") == "text")


def timestamp(value: str | None, tz: ZoneInfo) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt.replace(tzinfo=tz) if dt.tzinfo is None else dt.astimezone(tz)


def context_item(content: str, now: datetime) -> dict:
    return {"item_type": "UserMessageItem", "meta": {"item_id": uuid4().hex,
            "logical_turn_id": None, "timestamp": now.isoformat()},
            "parts": [{"type": "text", "text": content}]}


@dataclass
class Chat:
    index: int
    at: datetime
    message_id: str
    body: str
    own: bool


@dataclass
class Snapshot:
    chats: list[Chat]
    boundary: datetime
    previous_gap: float | None


def parse_chats(items: list[dict], tz: ZoneInfo) -> list[Chat]:
    chats = []
    for i, item in enumerate(items):
        content = text(item)
        match = MESSAGE.match(content)
        if item.get("item_type") != "UserMessageItem" or not match:
            continue
        attrs = dict(ATTR.findall(match.group(1)))
        at = timestamp(item.get("meta", {}).get("timestamp"), tz)
        if at is not None and attrs.get("msg_id"):
            chats.append(Chat(i, at, attrs["msg_id"], content[match.end():],
                              attrs.get("is_self_message") == "true"))
    return chats


def snapshot(items: list[dict], now: datetime, idle_minutes: int, recent_minutes: int) -> Snapshot:
    chats = parse_chats(items, now.tzinfo)
    valid = sorted((c for c in chats if c.at <= now + timedelta(minutes=5)), key=lambda c: (c.at, c.index))
    boundary = now - timedelta(minutes=recent_minutes)
    for before, after in zip(valid, valid[1:]):
        if (after.at - before.at).total_seconds() >= idle_minutes * 60:
            boundary = max(boundary, after.at)
    # 当前请求仍在读很久以前的历史时，不把旧话题继续视为当前一段。
    if valid and (now - valid[-1].at).total_seconds() >= idle_minutes * 60:
        boundary = max(boundary, now - timedelta(seconds=1))
    incoming = [c for c in valid if not c.own]
    gap = None
    if incoming:
        latest = incoming[-1]
        previous = [c for c in valid if c.at < latest.at]
        if previous:
            gap = (latest.at - previous[-1].at).total_seconds()
    return Snapshot(chats, boundary, gap)


def age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return "不到1分钟"
    if seconds < 3600:
        return f"{seconds // 60}分钟"
    if seconds < 86400:
        return f"{seconds // 3600}小时{seconds % 3600 // 60}分钟"
    return f"{seconds // 86400}天{seconds % 86400 // 3600}小时"


def singing_context(content: str, boundary: datetime) -> str:
    """终态任务不再每轮被描述成当前待办，实际任务仍可通过原工具查询。"""
    marker = content.find(SINGING)
    if marker < 0:
        return content
    start = content.find("\n[", marker)
    if start < 0:
        return content
    records, consumed = json.JSONDecoder().raw_decode(content[start + 1:])
    active = [r for r in records if r.get("status") not in {"success", "failed", "interrupted"}]
    recent = [r for r in records if r not in active and r.get("created_at", 0) >= boundary.timestamp()]
    fields = ("job_id", "song", "artist", "status", "sent", "sent_parts", "part_count", "created_at")
    compact = [{k: r[k] for k in fields if k in r} for r in active + recent]
    return (content[:marker] + "【学唱状态；后台资料】学唱工具可用。新的要求或重试调用generate_feiyu_song；仅同曲pending时查询或等待。"
            "口头唱一句通常指一段，默认连续2–3句，整首请求改唱副歌，较长自动分段。"
            "失败只代表那次没学顺，不是永久不会；发送成功以回执为准。没有当前唱歌话题时，不主动复述旧任务。\n"
            + json.dumps(compact, ensure_ascii=False) + content[start + 1 + consumed:])


def apply(items: list[dict], now: datetime, *, idle_minutes: int = 30, recent_minutes: int = 90,
          background_chars: int = 600, remembered: Snapshot | None = None,
          reply_message_id: str = "") -> tuple[list[dict], Snapshot, dict[str, Any]]:
    result = deepcopy(items)
    snap = snapshot(result, now, idle_minutes, recent_minutes)
    # Replyer 把自身历史转换为普通 assistant，生成时间不是原发送时间。
    # 优先以本轮 Planner 的真实消息记录匹配，重载空缓存时归属到前一条真实用户回合。
    assistant_times: dict[str, deque] = defaultdict(deque)
    if remembered:
        snap.boundary = max(snap.boundary, remembered.boundary)
        snap.previous_gap = remembered.previous_gap
        for chat in remembered.chats:
            if chat.own:
                assistant_times[chat.body.strip()].append(chat.at)
    chats_by_index = {c.index: c for c in snap.chats}
    dates: dict[int, datetime] = {}
    preceding_chat = None
    for i, item in enumerate(result):
        at = timestamp(item.get("meta", {}).get("timestamp"), now.tzinfo)
        if i in chats_by_index:
            preceding_chat = chats_by_index[i].at
        if item.get("item_type") == "AssistantMessageItem":
            known = assistant_times[text(item).strip()]
            at = known.popleft() if known else preceding_chat
        if at:
            dates[i] = at
    keep = set()
    old_summaries = []
    for i, item in enumerate(result):
        content = text(item)
        if content.startswith((CLOCK, BACKGROUND)):
            continue
        if content.startswith(SUMMARY):
            old_summaries.append((dates.get(i), content[len(SUMMARY):].strip()))
            continue
        # 系统人设、当前工具清单及当轮内部参考不按聊天年龄删除。
        if item.get("item_type") == "SystemMessageItem" or i not in dates or dates[i] >= snap.boundary:
            keep.add(i)
        if i in chats_by_index and chats_by_index[i].message_id == reply_message_id:
            keep.add(i)
    # 跨边界工具回合完整保留；避免只剩结果或丢失尚在运行的工具调用。
    groups: dict[str, set[int]] = defaultdict(set)
    calls, outputs = {}, set()
    for i, item in enumerate(result):
        if item.get("item_type") not in TOOL_TYPES:
            continue
        turn = item.get("meta", {}).get("logical_turn_id")
        if turn:
            groups["turn:" + turn].add(i)
        call = item.get("tool_call", {}).get("call_id") or item.get("call_id")
        if call:
            groups["call:" + call].add(i)
            if item["item_type"] == "FunctionCallItem":
                calls[call] = i
            else:
                outputs.add(call)
    keep.update(i for call, i in calls.items() if call not in outputs)
    changed = True
    while changed:
        old_count = len(keep)
        for indices in groups.values():
            if indices & keep:
                keep.update(indices)
        changed = len(keep) != old_count
    # 被保护的工具回合也带上实际发起它的用户消息，避免请求从孤立工具回合开始。
    for i in sorted(keep):
        if result[i].get("item_type") in TOOL_TYPES:
            anchors = [c.index for c in snap.chats if not c.own and c.index < i]
            if anchors:
                keep.add(anchors[-1])
    filtered = []
    for i, item in enumerate(result):
        if i not in keep:
            continue
        content = text(item)
        for part in item.get("parts", []):
            if part.get("type") == "text" and SINGING in part.get("text", ""):
                part["text"] = singing_context(part["text"], snap.boundary)
        if i in dates and (i in chats_by_index or item.get("item_type") == "AssistantMessageItem"):
            stamp = dates[i].strftime("%Y-%m-%d %H:%M:%S")
            item["parts"].insert(0, {"type": "text", "text": f"[发言时间：{stamp}]"})
        filtered.append(item)
    if background_chars and old_summaries:
        at, content = old_summaries[-1]
        content = content.replace("当前话题", "当时的话题").replace("未完事项", "当时记录的事项（状态可能已变）")
        stamp = at.strftime("%Y-%m-%d %H:%M") if at else "时间不详"
        leading_systems = 0
        while leading_systems < len(filtered) and filtered[leading_systems].get("item_type") == "SystemMessageItem":
            leading_systems += 1
        filtered.insert(leading_systems,
            context_item(f"{BACKGROUND}；整理于{stamp}】以下是旧资料，不能据此认定当前仍在做或尚未完成。\n{content[:background_chars]}", now))
    incoming = [c for c in snap.chats if not c.own]
    last = incoming[-1] if incoming else None
    lines = [CLOCK, f"现在：{now:%Y-%m-%d %H:%M:%S}，星期{'一二三四五六日'[now.weekday()]}（{now.tzinfo}）。",
             f"当前这段原文从{snap.boundary:%m-%d %H:%M}起；更早聊天仍在本地原记录中，是否已生成长期记忆以实际检索为准。"]
    if last:
        lines.append(f"本会话最后一次来信：{last.at:%m-%d %H:%M}，距现在{age((now-last.at).total_seconds())}。")
    if snap.previous_gap is not None:
        lines.append(f"这次来信与此前真实交谈相隔{age(snap.previous_gap)}。")
    lines.append("顺着最新来信说话。久别后自然重新碰面，只有对方接着提起才续旧话题；不挨个结算旧任务。"
                 "旧摘要和召回是背景，当前状态以新消息及实际工具结果为准。时间间隔只属于这个会话，不代表期间没有和别人聊天。"
                 "生活模拟是角色生活，读网页、玩游戏、学歌等真实行动须有工具记录；不把小传当刚发生的经历。")
    lines.append("对方回顾上午、下午或往事时，先依据聊天记忆及行动记录；当前生活状态不能证明过去在做什么。"
                 "没加载的记录不等于没发生，不用模拟睡觉否认实际交谈；记忆不足时用query_memory查询。")
    filtered.append(context_item("\n".join(lines), now))
    return filtered, snap, {"input_items": len(items), "output_items": len(filtered),
                            "removed_items": len(result)-len(keep), "boundary": snap.boundary.isoformat()}
