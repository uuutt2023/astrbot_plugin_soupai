"""判断后端分发的接线测试（main.py）。

重点验证那条安全底线：**Jev 出任何问题都必须回落到 LLM，玩家永远拿得到
「是/否/是也不是/不重要」四档之一**，游戏不会因为 Jev 挂掉。
"""

import pytest


class FakeJevClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def evaluate(self, cfg, state, questions, idem=None):
        self.calls.append({"state": state, "questions": questions})
        return self.result


def jev_ready(choice="yes", confidence=0.93):
    # 概率质量要落在被选中的那一档上——传输层就是这么返回的
    probabilities = {key: 0.0 for key in ("yes", "no", "partial", "irrelevant")}
    probabilities[choice] = confidence
    return {
        "status": "ready",
        "answers": {
            "verdict": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        },
        "seconds": 2.1,
        "model": "jev-latest",
        "usage": {"input_tokens": 88},
    }


JEV_CONFIG = {
    "judge_backend": "jev",
    "jev_judge": {
        "enabled": True,
        "api_key": "sk-test",
        "base_url": "https://jevmodel.org",
        "model": "jev-latest",
        "gate_mode": "threshold",
        "min_confidence": 0.6,
    },
}


class TestBackendDispatch:
    @pytest.mark.asyncio
    async def test_llm_backend_never_calls_jev(self, make_plugin):
        plugin = make_plugin({"judge_backend": "llm"})
        plugin.jev_client = FakeJevClient(jev_ready())

        reply = await plugin.judge_question("死者是不是男的", "他是男性")

        assert reply == "是"
        assert plugin.jev_client.calls == []
        assert len(plugin._test_provider_calls) == 1

    @pytest.mark.asyncio
    async def test_jev_backend_uses_jev_and_skips_llm(self, make_plugin):
        plugin = make_plugin(JEV_CONFIG)
        plugin.jev_client = FakeJevClient(jev_ready(choice="no"))

        reply = await plugin.judge_question("死者是不是男的", "他是男性")

        assert reply == "否"
        assert len(plugin.jev_client.calls) == 1
        assert plugin._test_provider_calls == []

    @pytest.mark.asyncio
    async def test_jev_backend_falls_back_when_disabled(self, make_plugin):
        config = {**JEV_CONFIG, "jev_judge": {**JEV_CONFIG["jev_judge"], "enabled": False}}
        plugin = make_plugin(config)
        plugin.jev_client = FakeJevClient(jev_ready())

        reply = await plugin.judge_question("死者是不是男的", "他是男性")

        assert reply == "是"  # LLM 的答案
        assert plugin.jev_client.calls == []
        assert len(plugin._test_provider_calls) == 1

    @pytest.mark.asyncio
    async def test_jev_backend_falls_back_on_transport_error(self, make_plugin):
        plugin = make_plugin(JEV_CONFIG)
        plugin.jev_client = FakeJevClient({"status": "upstream_error"})

        reply = await plugin.judge_question("死者是不是男的", "他是男性")

        assert reply == "是"
        assert len(plugin._test_provider_calls) == 1

    @pytest.mark.asyncio
    async def test_jev_backend_falls_back_on_exception(self, make_plugin):
        class ExplodingClient:
            async def evaluate(self, *args, **kwargs):
                raise RuntimeError("boom")

        plugin = make_plugin(JEV_CONFIG)
        plugin.jev_client = ExplodingClient()

        reply = await plugin.judge_question("死者是不是男的", "他是男性")

        assert reply == "是"
        assert len(plugin._test_provider_calls) == 1

    @pytest.mark.asyncio
    async def test_hybrid_uses_jev_when_confident(self, make_plugin):
        config = {**JEV_CONFIG, "judge_backend": "hybrid"}
        plugin = make_plugin(config)
        plugin.jev_client = FakeJevClient(jev_ready(choice="partial"))

        reply = await plugin.judge_question("死者是不是男的", "他是男性")

        assert reply == "是也不是"
        assert plugin._test_provider_calls == []

    @pytest.mark.asyncio
    async def test_hybrid_falls_back_when_unconfident(self, make_plugin):
        config = {**JEV_CONFIG, "judge_backend": "hybrid"}
        plugin = make_plugin(config)
        plugin.jev_client = FakeJevClient(jev_ready(confidence=0.2))

        reply = await plugin.judge_question("死者是不是男的", "他是男性")

        assert reply == "是"  # LLM 的答案
        assert len(plugin.jev_client.calls) == 1
        assert len(plugin._test_provider_calls) == 1

    @pytest.mark.asyncio
    async def test_invalid_backend_falls_back_to_llm(self, make_plugin):
        plugin = make_plugin({"judge_backend": "nonexistent"})
        plugin.jev_client = FakeJevClient(jev_ready())

        reply = await plugin.judge_question("死者是不是男的", "他是男性")

        assert reply == "是"
        assert plugin.jev_client.calls == []
        assert len(plugin._test_provider_calls) == 1

    @pytest.mark.asyncio
    async def test_jev_receives_puzzle_and_history(self, make_plugin):
        plugin = make_plugin(JEV_CONFIG)
        plugin.jev_client = FakeJevClient(jev_ready())

        await plugin.judge_question(
            "死者是不是男的",
            "他是男性",
            puzzle="深夜的浴室",
            qa_history=[{"question": "他是不是自杀", "answer": "否"}],
        )

        state = plugin.jev_client.calls[0]["state"]
        assert "深夜的浴室" in state
        assert "他是不是自杀" in state
        assert "死者是不是男的" in state


class TestTerminate:
    @pytest.mark.asyncio
    async def test_terminate_closes_jev_client(self, make_plugin):
        closed = []

        class ClosableClient:
            async def close(self):
                closed.append(True)

        plugin = make_plugin(JEV_CONFIG)
        plugin.jev_client = ClosableClient()

        await plugin.terminate()

        assert closed == [True]
