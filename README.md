# 肥鱼时间上下文

## 本地修改说明（2026-11-24）

按用户实际发言间隔划分会话，内部心跳和工具轮询不会延长这一段。按最新来信判断话题，往次摘要带日期作为背景。实现见 `temporal.py`、`focus.py` 和 `recall.py`。

可由模型决定每段对话要记什么、何时再整理，并保存长期摘要和人物印象。实现见 `segment_memory.py` 和 `memory_client.py`。此版本维护 MaiBot 1.3 系列；11 月 24 日核对时，公开版与本地功能代码一致。

对应功能提交 `3135d92`。[公开版本 v1.3.0](https://github.com/BRXYM/maibot-feiyu-time-context/releases/tag/v1.3.0)

按真实聊天时间分段与整理记忆的 MaiBot 1.3 插件。

## 安装与使用

解压 Release 到麦麦的 `plugins/feiyu-time-context/`，目录内应直接包含 `_manifest.json` 和 `plugin.py`。

此版本面向 MaiBot 1.3.19。时间分段与可选的记忆整理可分别配置。使用记忆写入前在本机 config.toml 填写机器人 QQ 及本机 WebUI 地址，先备份记忆。麦麦 1.4 有原生上下文和记忆功能，本插件仍作为 1.3 兼容版本维护。

## 许可证

插件代码采用 MIT。第三方运行时及模型采用各自许可。
