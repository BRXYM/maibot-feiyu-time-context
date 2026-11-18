"""通过宿主公开 WebUI 接口写入原生记忆；不直接改动运行中的记忆数据库。"""
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
import asyncio
import hashlib
import json
import re


class MissingImportTask(RuntimeError):
    """宿主重启后，内存中的导入队列不再包含这个任务。"""


def paragraph_hash(content: str) -> str:
    normalized = content.strip().replace('\r\n', '\n').replace('\r', '\n')
    normalized = re.sub(r'[^\S\n]+', ' ', normalized)
    return hashlib.sha256(normalized.encode()).hexdigest()


class NativeMemoryClient:
    def __init__(self, base_url: str, root: Path, recent_days: int = 7):
        parsed = urlsplit(base_url)
        if parsed.scheme not in {'http', 'https'} or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('记忆写入仅连接本机麦麦，不能把网页登录凭据发送到外部地址')
        self.base = base_url.rstrip('/') + '/api/webui/memory'
        self.root = root
        self.recent_days = recent_days

    def _request(self, method, path, body=None, timeout=25):
        token = json.loads((self.root/'data/webui.json').read_text(encoding='utf-8'))['access_token']
        req = Request(self.base+path, data=json.dumps(body, ensure_ascii=False).encode() if body is not None else None,
                      headers={'Cookie': 'maibot_session='+token, 'Content-Type': 'application/json'}, method=method)
        with urlopen(req, timeout=timeout) as response:
            result = json.load(response)
        if result.get('success') is False:
            raise RuntimeError(str(result.get('error') or result.get('detail') or '原生记忆接口失败'))
        return result

    async def request(self, method, path, body=None, timeout=25):
        return await asyncio.to_thread(self._request, method, path, body, timeout)

    async def authorize_streams(self, streams):
        result = await self.request('GET', '/config')
        config = result['config']
        filtering = config['filter']
        changed = False
        if filtering.get('enabled') and filtering.get('mode') == 'whitelist':
            patterns = list(filtering.get('chats', []))
            for stream in streams:
                token = 'stream:'+stream['session_id']
                if token not in patterns:
                    patterns.append(token)
                    changed = True
            filtering['chats'] = patterns
        # 用 AI 决定整理时机，停止固定每36条再写一遍摘要的旧通道。
        if config['integration'].get('chat_summary_writeback_enabled'):
            config['integration']['chat_summary_writeback_enabled'] = False
            changed = True
        if changed:
            await self.request('PUT', '/config', {'config': config})
        return changed

    async def profile(self, user_id):
        result = await self.request('GET', '/profiles/search?'+urlencode({'platform': 'qq', 'user_id': user_id, 'limit': 1}))
        items = result.get('items', [])
        if not items:
            raise ValueError('该真实聊天对象尚未建立原生人物身份')
        return items[0]

    async def recall(self, session, query, limit=6):
        result = await self.request('GET', '/query/aggregate?'+urlencode({
            'query': query, 'chat_id': session, 'limit': limit}), timeout=3)
        return result.get('hits', [])

    async def has_paragraph(self, content, session):
        try:
            data = await self.request('GET', '/graph/paragraph-detail?'+urlencode({'paragraph_hash': paragraph_hash(content)}))
        except HTTPError as exc:
            if exc.code == 404:
                return False
            raise
        paragraph = data.get('paragraph') or {}
        return paragraph.get('hash') == paragraph_hash(content) and paragraph.get('source') == 'chat_summary:'+session

    async def submit_summary(self, segment):
        # 宿主按精确的chat_summary:<真实流ID>识别来源；分段编号已经保存在正文和本地日志中。
        unit = {'content': segment['content'], 'source': 'chat_summary:'+segment['session'],
                'knowledge_type': 'narrative', 'person_ids': [],
                'time_meta': {'event_time_start': segment['start_time'], 'event_time_end': segment['end_time'],
                              'time_confidence': 1.0}}
        result = await self.request('POST', '/import/paste', {
            'name': 'feiyu-segment-'+segment['id']+'.json', 'input_mode': 'json',
            'content': json.dumps({'paragraphs': [unit]}, ensure_ascii=False),
            'scope_type': 'chat', 'chat_id': segment['session'], 'llm_enabled': False,
            'dedupe_policy': 'content_hash', 'file_concurrency': 1, 'chunk_concurrency': 1})
        task = result.get('task', result)
        if not task.get('task_id'):
            raise ValueError('原生导入未返回任务编号')
        return task['task_id']

    async def task_status(self, task_id):
        try:
            result = await self.request('GET', '/import/tasks/'+task_id)
        except HTTPError as error:
            if error.code == 404:
                raise MissingImportTask(task_id) from error
            raise
        except RuntimeError as error:
            if str(error) == '任务不存在':
                raise MissingImportTask(task_id) from error
            raise
        task = result.get('task', result)
        return task

    async def retry_task(self, task_id):
        result = await self.request('POST', '/import/tasks/'+task_id+'/retry', {})
        return result.get('task', result).get('task_id', task_id)

    async def write_person_update(self, person_id, update, segment):
        reason = json.dumps({'plugin': 'feiyu-segment-memory', 'segment_id': segment['id'],
                             'chat_id': segment['session'], 'evidence': update['evidence']}, ensure_ascii=False)
        payload = {'scope_type': 'person', 'scope_id': person_id, 'fact_key': update['key'],
                   'value_text': update['text'], 'cardinality': 'set', 'polarity': 'positive',
                   'stability': update['stability'], 'profile_section': update['section'],
                   'authority': update['authority'], 'confidence': update['confidence'],
                   'valid_from': segment['end_time'], 'reason': reason, 'updated_by': 'local.zz-feiyu-time'}
        if update['stability'] == 'temporal':
            payload['valid_to'] = segment['end_time']+self.recent_days*86400
        result = await self.request('POST', '/facts', payload)
        claim = result.get('claim') or {}
        if not claim.get('claim_id'):
            raise ValueError('人物事实未返回实际入库编号')
        verified = await self.request('GET', '/facts/'+claim['claim_id'])
        actual = verified.get('claim', verified)
        if actual.get('scope_id') != person_id or actual.get('value_text') != update['text']:
            raise ValueError('人物事实入库后核对不一致')
        return claim['claim_id']

    async def refresh_profile(self, person_id):
        result = await self.request('GET', '/profiles/query?'+urlencode({'person_id': person_id, 'force_refresh': 'true'}), timeout=90)
        if result.get('person_id') != person_id:
            raise ValueError('人物画像刷新对象不一致')
        return result
