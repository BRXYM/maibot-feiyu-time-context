from maibot_sdk import Field, PluginConfigBase


class TimeSection(PluginConfigBase):
    __ui_label__ = "时间与当前对话"
    config_version: str = Field(default="1.0.0", description="配置版本")
    enabled: bool = Field(default=True, description="启用时间感知和请求上下文分段")
    timezone: str = Field(default="Asia/Shanghai", description="使用 IANA 时区")
    idle_minutes: int = Field(default=30, ge=5, le=1440, description="真实聊天间隔达到多少分钟后开始新一段；内部心跳不续期")
    recent_minutes: int = Field(default=90, ge=10, le=1440, description="当前请求最多保留多久的聊天原文；长期记录不删除")
    background_chars: int = Field(default=600, ge=0, le=2000, description="旧滚动摘要转成历史资料后的最大字数；0 表示不随请求附带")


class MemorySection(PluginConfigBase):
    __ui_label__ = "AI对话整理与人物印象"
    enabled: bool = Field(default=True, description="AI决定分段、记忆内容及下次整理时间")
    base_url: str = Field(default="http://127.0.0.1:8001", description="本机麦麦WebUI地址")
    model_task: str = Field(default="utils", description="用于整理的宿主模型任务或已配置模型名")
    bot_id: str = Field(default="", description="只处理此机器人已存在的QQ私聊流")
    bootstrap_hours: int = Field(default=24, ge=0, le=168, description="首次启用回看多少小时，后续从持久化游标接着处理")
    max_reviews_per_hour: int = Field(default=16, ge=1, le=120, description="模型调用上限；实际整理时机由AI决定")
    recent_days: int = Field(default=7, ge=1, le=90, description="临时偏好和近期互动在画像中的最长有效天数；长期摘要继续保留")
    recall_enabled: bool = Field(default=True, description="新来信自动检索往次对话，规划器和回复器共用正文")
    recall_limit: int = Field(default=3, ge=1, le=5, description="自动召回最多几段同一聊天流的历史摘要")
    recall_chars: int = Field(default=1800, ge=600, le=4000, description="自动召回正文预算；先去除内部编号等元数据")


class TimeConfig(PluginConfigBase):
    plugin: TimeSection = Field(default_factory=TimeSection)
    memory: MemorySection = Field(default_factory=MemorySection)
