"""pytest 公共装置。

这个仓库原本没有测试框架。为了能测「判断后端分发」这段接线逻辑，
这里用最小 AstrBot 桩件把 main.py 导入进来——桩件只提供插件真正用到的
那几个名字，不追求还原框架行为。

运行：``pytest tests/ -v``（需要 pytest；插件运行时本身不依赖它）
"""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ── 最小 AstrBot 桩件 ──────────────────────────────────────────────────
def _install_astrbot_stub():
    if "astrbot" in sys.modules:
        return

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    event = types.ModuleType("astrbot.api.event")
    star = types.ModuleType("astrbot.api.star")
    provider = types.ModuleType("astrbot.api.provider")
    components = types.ModuleType("astrbot.api.message_components")
    core = types.ModuleType("astrbot.core")
    core_utils = types.ModuleType("astrbot.core.utils")
    session_waiter = types.ModuleType("astrbot.core.utils.session_waiter")

    class _Any:
        """万能占位：支持属性访问与调用（调用返回装饰器）。"""

        def __getattr__(self, name):
            return _Any()

        def __call__(self, *args, **kwargs):
            def deco(func):
                return func

            if len(args) == 1 and not kwargs and callable(args[0]):
                return args[0]
            return deco

    class _Filter:
        PermissionType = _Any()

        def command(self, *args, **kwargs):
            def deco(func):
                func.__filter_cmd__ = True
                return func

            if len(args) == 1 and not kwargs and callable(args[0]):
                return args[0]
            return deco

        def __getattr__(self, name):
            return _Any()

    class AstrMessageEvent:
        pass

    class MessageEventResult:
        pass

    class Context:
        pass

    class Star:
        def __init__(self, context=None, *args, **kwargs):
            self.context = context

    class StarTools:
        @staticmethod
        def get_data_dir():
            return Path("/tmp/soupai_test_data")

    def register(*args, **kwargs):
        def deco(cls):
            return cls

        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return deco

    class LLMResponse:
        def __init__(self, completion_text=""):
            self.completion_text = completion_text

    class At:
        pass

    class SessionController:
        pass

    class SessionFilter:
        def __init__(self, *args, **kwargs):
            pass

    def session_waiter(*args, **kwargs):
        def deco(func):
            return func

        return deco

    event.filter = _Filter()
    event.AstrMessageEvent = AstrMessageEvent
    event.MessageEventResult = MessageEventResult
    star.Context = Context
    star.Star = Star
    star.StarTools = StarTools
    star.register = register
    provider.LLMResponse = LLMResponse
    components.At = At
    session_waiter.session_waiter = session_waiter
    session_waiter.SessionController = SessionController
    session_waiter.SessionFilter = SessionFilter

    api.event = event
    api.star = star
    api.provider = provider
    api.message_components = components
    api.logger = __import__("logging").getLogger("astrbot.stub")
    api.AstrBotConfig = dict
    astrbot.api = api
    astrbot.core = core
    core.utils = core_utils
    core_utils.session_waiter = session_waiter

    sys.modules.update(
        {
            "astrbot": astrbot,
            "astrbot.api": api,
            "astrbot.api.event": event,
            "astrbot.api.star": star,
            "astrbot.api.provider": provider,
            "astrbot.api.message_components": components,
            "astrbot.core": core,
            "astrbot.core.utils": core_utils,
            "astrbot.core.utils.session_waiter": session_waiter,
        }
    )


_install_astrbot_stub()


# 插件以包方式导入：把仓库根目录的父目录放进 sys.path，
# 让 `astrbot_plugin_soupai.main` 成为一个可导入的包。
_PARENT = str(ROOT.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

PLUGIN_PKG = ROOT.name


@pytest.fixture(scope="session")
def plugin_module():
    """导入 main.py（此时 astrbot 已被桩件顶替）。"""
    import importlib

    return importlib.import_module(f"{PLUGIN_PKG}.main")


@pytest.fixture
def make_plugin(plugin_module):
    """造一个带假 provider 的插件实例。"""

    def _make(config=None, judge_text="是"):
        from astrbot.api.provider import LLMResponse

        calls = []

        class FakeProvider:
            async def text_chat(self, prompt=None, contexts=None, func_tool=None,
                                image_urls=None, system_prompt=None, **kwargs):
                calls.append({"prompt": prompt, "system_prompt": system_prompt})
                return LLMResponse(completion_text=judge_text)

        class FakeContext:
            def __init__(self):
                self.provider = FakeProvider()

            def get_using_provider(self):
                return self.provider

            def get_provider_by_id(self, provider_id):
                return self.provider

        cfg = dict(config or {})
        instance = plugin_module.SoupaiPlugin(FakeContext(), cfg)
        instance.data_path = Path("/tmp/soupai_test_data")
        instance.data_path.mkdir(parents=True, exist_ok=True)
        instance._test_provider_calls = calls
        return instance

    return _make
