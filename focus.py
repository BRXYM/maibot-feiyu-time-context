"""让当前来信决定话题，历史仍可读取但不再充当新一轮待办。"""
from datetime import datetime
import re

from .temporal import context_item, parse_chats, text, timestamp

MARKER = '【本轮话题焦点】'
POLICY_MARKER = '【话题跟随规则】'
CURRENT = '【当前来信；请结合最近对话理解】'
POLICY = (POLICY_MARKER + '\n'
          '由最新真实来信的语义决定当前话题，不以聊天间隔或旧摘要决定。对方提出新的独立内容，就自然接新内容；'
          '此前的任务、提议、疑问和玩笑可以暂时搁下，不自动重新提起、汇报、催问或继续推销。'
          '不要把自己上一轮的提议当成用户已经答应的任务，也不靠反复提旧事维持亲近感。'
          '只有对方在追问、引用、指代前文或主动回顾时，才接相应的旧话题；“还有吗”“那张呢”等短追问要结合紧邻对话理解。'
          '历史记忆是可用背景，只在与当前内容直接相关时引用，检索命中不代表应该说出来。'
          '回复方向建议若还在追上一话题，按最新来信调整；角色性格保留，不把每个话题都变成必须结算的任务。'
          '真正到达的后台完成通知可简短交付结果，不顺带重开旧问答。若最新来信已答完且没有新信息，就自然等待，不重复补充或追问。')

# 这里只识别明确的承接表达以选择检索查询，不按词表裁剪真实聊天，也不替模型判定所有话题。
FOLLOWUP = re.compile(
    r'^(?:还有(?:吗|呢|别的|其他)|然后呢|后来呢|接着呢|继续(?:吧|说|讲|找|发|唱)?[啊吧呢]?|'
    r'(?:那|这)(?:张|个|首|段|件|位|次|句|里面)|'
    r'(?:今天|昨天)?(?:上午|下午|晚上|中午|早上)呢|'
    r'(?:那)?你?(?:就)?按(?:这个|那个|刚才)|'
    r'你(?:刚才|之前|上次)说|再(?:来|发|找|唱|试)(?:一|两|几|点|些|张|遍))'
)
NEW_TOPIC = re.compile(r'^(?:对了[，,。\s]|换(?:个|一个)话题|说点别的|先不聊|不聊这个|不说这个|这个先放一边)')
ABANDONED_PREFIX = re.compile(r'^.{0,45}?(?:先不聊.{0,20}?了|不聊这个了|不说这个了|先放一边|先不提了|先不找了)[，,。；;\s]+')


def without_time_labels(items):
    return [{**item, 'parts': [part for part in item.get('parts', []) if not
             (part.get('type') == 'text' and part.get('text', '').startswith('[发言时间：'))]}
            for item in items]


def is_followup(body):
    body = body.strip()
    return not NEW_TOPIC.match(body) and bool(FOLLOWUP.match(body) or body.startswith('[回复了'))


def recall_query(incoming):
    latest = incoming[-1].body.strip()
    if is_followup(latest):
        return '\n'.join(chat.body.strip()[-300:] for chat in incoming[-3:])
    # 已明确放下的前半句不再作为向量查询里的主题词；原消息仍完整提供给模型。
    current = ABANDONED_PREFIX.sub('', latest).strip() or latest
    return current[-600:]


def newer_internal_trigger(items, latest, now):
    for item in items:
        if text(item).lstrip().startswith(('<plugin_proactive_task', '<subagent_result')):
            at = timestamp(item.get('meta', {}).get('timestamp'), now.tzinfo)
            if at and at >= latest.at:
                return True
    return False


def apply_focus(items, now: datetime, remembered=None):
    plain = without_time_labels(items)
    incoming = [chat for chat in parse_chats(plain, now.tzinfo) if not chat.own and chat.at <= now]
    if not incoming and remembered:
        incoming = [chat for chat in remembered.chats if not chat.own and chat.at <= now]
    if not incoming:
        return items
    latest = max(incoming, key=lambda chat: (chat.at, chat.index))
    # 新的内部主动聊天任务没有新的用户来信，交由它自己的任务调度，不拿旧来信覆盖它。
    if newer_internal_trigger(items, latest, now):
        return [item for item in items if not text(item).startswith((MARKER, POLICY_MARKER, CURRENT))]
    kept = [item for item in items if not text(item).startswith((MARKER, POLICY_MARKER, CURRENT))]
    policy = context_item(POLICY, now)
    policy['item_type'] = 'SystemMessageItem'
    leading = 0
    while leading < len(kept) and kept[leading].get('item_type') == 'SystemMessageItem':
        leading += 1
    kept.insert(leading, policy)
    mode = ('这是一句承接表达，结合紧邻对话理解指代；不要跳到无关的往次记忆。' if is_followup(latest.body) else
            '先理解这条来信本身；若主题已变，直接接新主题，不附带上一话题的进度、玩笑或追问。')
    focus = context_item(MARKER + '\n' + mode + '\n当前真实来信：\n' + latest.body.strip()[:1800], now)
    kept.append(focus)
    return kept
