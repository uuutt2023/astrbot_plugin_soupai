"""Jev 判断层（core/jev_judge.py）的纯逻辑测试。

这一层不 import astrbot，所以测试不需要框架环境。
"""

import importlib
import random
import sys
from pathlib import Path

import pytest

# 插件目录名可能不是 astrbot_plugin_soupai（测试时会被 clone 成别的名字），
# 这里按实际目录名解析包路径，避免硬编码。
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))
jev_judge = importlib.import_module(f"{_ROOT.name}.core.jev_judge")


# ── 请求体形状 ──────────────────────────────────────────────────────────
class TestQuestions:
    def test_single_choice_question(self):
        questions = jev_judge.build_questions()
        assert list(questions) == [jev_judge.QUESTION_NAME]
        question = questions[jev_judge.QUESTION_NAME]
        assert question["type"] == "choice"
        assert question["instructions"].strip()
        assert set(question["criteria"]) == set(jev_judge.VERDICT_LABELS)

    def test_criteria_keys_match_verdict_labels(self):
        criteria = jev_judge.build_questions()[jev_judge.QUESTION_NAME]["criteria"]
        assert set(criteria) == {"yes", "no", "partial", "irrelevant"}

    def test_instructions_within_documented_limit(self):
        instructions = jev_judge.build_questions()[jev_judge.QUESTION_NAME]["instructions"]
        assert len(instructions) <= 1800  # 文档上限

    def test_criteria_are_english(self):
        """Jev 主训练语言是英文，选项描述里不该混中文。"""
        criteria = jev_judge.build_questions()[jev_judge.QUESTION_NAME]["criteria"]
        for key, text in criteria.items():
            assert not any("一" <= ch <= "鿿" for ch in text), key


# ── state 构造 ──────────────────────────────────────────────────────────
class TestBuildState:
    PUZZLE = "深夜停电，男主人死在上锁的浴室里。"
    TRUTH = "他是在浴缸里溺死的，滑倒时撞到了头，锁是插销自己落下的。"
    QUESTION = "死者是不是滑倒的"

    def test_contains_all_parts(self):
        state = jev_judge.build_state(self.PUZZLE, self.TRUTH, self.QUESTION)
        assert "[Scenario]" in state
        assert "[Truth]" in state
        assert "[Player]" in state
        assert self.TRUTH in state
        assert self.QUESTION in state

    def test_no_history_section_when_empty(self):
        state = jev_judge.build_state(self.PUZZLE, self.TRUTH, self.QUESTION, [])
        assert "[History]" not in state

    def test_history_rounds_limits_entries(self):
        history = [
            {"question": f"问题{i}", "answer": "是"} for i in range(10)
        ]
        state = jev_judge.build_state(
            self.PUZZLE, self.TRUTH, self.QUESTION, history, history_rounds=3
        )
        assert "问题0" not in state
        assert "问题9" in state
        assert state.count("Q: ") == 3

    def test_history_rounds_zero_disables_history(self):
        history = [{"question": "问题0", "answer": "是"}]
        state = jev_judge.build_state(
            self.PUZZLE, self.TRUTH, self.QUESTION, history, history_rounds=0
        )
        assert "[History]" not in state

    def test_overlong_state_drops_oldest_history_first(self):
        history = [{"question": f"很长的历史问题{'x' * 200}{i}", "answer": "否"} for i in range(6)]
        state = jev_judge.build_state(
            self.PUZZLE,
            self.TRUTH,
            self.QUESTION,
            history,
            max_chars=600,
            history_rounds=6,
        )
        assert len(state) <= 600
        # 真相和玩家提问必须还在（判断的必需信息）
        assert self.TRUTH in state
        assert self.QUESTION in state

    def test_extreme_overflow_is_truncated(self):
        state = jev_judge.build_state(
            "题面" * 500, "真相" * 500, "提问", None, max_chars=300
        )
        assert len(state) <= 300


# ── 答案解析 ────────────────────────────────────────────────────────────
class TestVerdictConfidence:
    def test_prefers_probability_of_selected_choice(self):
        answer = {
            "type": "choice",
            "choice": "no",
            "confidence": 0.9,
            "probabilities": {"yes": 0.05, "no": 0.9, "partial": 0.03, "irrelevant": 0.02},
        }
        choice, confidence = jev_judge.verdict_confidence(answer)
        assert choice == "no"
        assert confidence == pytest.approx(0.9)

    def test_falls_back_to_confidence_field(self):
        choice, confidence = jev_judge.verdict_confidence(
            {"type": "choice", "choice": "yes", "confidence": 0.42}
        )
        assert choice == "yes"
        assert confidence == pytest.approx(0.42)

    def test_unknown_choice_is_rejected(self):
        choice, confidence = jev_judge.verdict_confidence(
            {"type": "choice", "choice": "maybe", "confidence": 0.9}
        )
        assert choice is None
        assert confidence is None

    def test_non_dict_is_rejected(self):
        assert jev_judge.verdict_confidence(None) == (None, None)
        assert jev_judge.verdict_confidence("yes") == (None, None)


# ── 门控 ────────────────────────────────────────────────────────────────
class TestDecideGate:
    def test_threshold_accepts_at_and_above(self):
        assert jev_judge.decide_gate(0.6, 0.6, "threshold")[0] is True
        assert jev_judge.decide_gate(0.61, 0.6, "threshold")[0] is True
        assert jev_judge.decide_gate(0.59, 0.6, "threshold")[0] is False

    def test_missing_probability_is_rejected(self):
        assert jev_judge.decide_gate(None, 0.6, "threshold") == (False, 0.0)

    def test_sample_is_deterministic_with_seeded_rng(self):
        rng = random.Random(42)
        first = jev_judge.decide_gate(0.5, 0.6, "sample", rng)
        rng2 = random.Random(42)
        second = jev_judge.decide_gate(0.5, 0.6, "sample", rng2)
        assert first == second
        assert first[0] is (first[1] <= 0.5)

    def test_sample_distribution_tracks_probability(self):
        rng = random.Random(7)
        hits = sum(
            1 for _ in range(2000) if jev_judge.decide_gate(0.3, 0.6, "sample", rng)[0]
        )
        assert 0.25 < hits / 2000 < 0.35

    def test_sample_always_accepts_probability_one(self):
        rng = random.Random(1)
        for _ in range(50):
            assert jev_judge.decide_gate(1.0, 0.6, "sample", rng)[0] is True


# ── 判定映射 ────────────────────────────────────────────────────────────
class TestPickVerdict:
    def _answer(self, key, confidence):
        return {
            "type": "choice",
            "choice": key,
            "confidence": confidence,
            "probabilities": {k: (confidence if k == key else 0.0) for k in jev_judge.VERDICT_LABELS},
        }

    @pytest.mark.parametrize(
        "key,label",
        [
            ("yes", "是"),
            ("no", "否"),
            ("partial", "是也不是"),
            ("irrelevant", "不重要"),
        ],
    )
    def test_labels(self, key, label):
        picked = jev_judge.pick_verdict(self._answer(key, 0.95), 0.6, "threshold")
        assert picked["ok"] is True
        assert picked["accepted"] is True
        assert picked["verdict"] == label

    def test_low_confidence_falls_back(self):
        picked = jev_judge.pick_verdict(self._answer("yes", 0.2), 0.6, "threshold")
        assert picked["ok"] is False
        assert picked["accepted"] is False
        assert picked["verdict"] == "是"  # 档位仍然解析出来了，只是没采信

    def test_invalid_answer_is_not_ok(self):
        picked = jev_judge.pick_verdict({"type": "choice", "choice": "nope"}, 0.6, "threshold")
        assert picked["ok"] is False
        assert picked["verdict"] == ""


# ── 端到端（假客户端）──────────────────────────────────────────────────
class FakeClient:
    """记录请求、返回预设结果的假 Jev 客户端。"""

    def __init__(self, result):
        self.result = result
        self.calls = []

    async def evaluate(self, cfg, state, questions, idem=None):
        self.calls.append({"cfg": cfg, "state": state, "questions": questions})
        return self.result


class TestJudgeViaJev:
    PUZZLE = "题面"
    TRUTH = "真相"
    QUESTION = "问题"

    @pytest.mark.asyncio
    async def test_ready_and_confident(self):
        client = FakeClient(
            {
                "status": "ready",
                "answers": {
                    "verdict": {
                        "type": "choice",
                        "choice": "no",
                        "confidence": 0.93,
                        "probabilities": {"yes": 0.02, "no": 0.93},
                    }
                },
                "seconds": 2.4,
                "model": "jev-latest",
                "usage": {"input_tokens": 120},
            }
        )
        result = await jev_judge.judge_via_jev(
            client, {}, self.PUZZLE, self.TRUTH, self.QUESTION
        )
        assert result["ok"] is True
        assert result["verdict"] == "否"
        assert result["confidence"] == pytest.approx(0.93)
        assert result["state_chars"] > 0

    @pytest.mark.asyncio
    async def test_ready_but_unconfident_falls_back(self):
        client = FakeClient(
            {
                "status": "ready",
                "answers": {
                    "verdict": {
                        "type": "choice",
                        "choice": "yes",
                        "confidence": 0.2,
                        "probabilities": {"yes": 0.2},
                    }
                },
            }
        )
        result = await jev_judge.judge_via_jev(
            client, {}, self.PUZZLE, self.TRUTH, self.QUESTION
        )
        assert result["ok"] is False
        assert "置信度不足" in result["reason"]

    @pytest.mark.asyncio
    async def test_transport_failure_is_reported_with_reason(self):
        client = FakeClient({"status": "upstream_error", "error_message": "boom"})
        result = await jev_judge.judge_via_jev(
            client, {}, self.PUZZLE, self.TRUTH, self.QUESTION
        )
        assert result["ok"] is False
        assert result["reason"] == "Jev 上游故障"

    @pytest.mark.asyncio
    async def test_unconfigured_is_reported(self):
        client = FakeClient({"status": "unconfigured"})
        result = await jev_judge.judge_via_jev(
            client, {}, self.PUZZLE, self.TRUTH, self.QUESTION
        )
        assert result["ok"] is False
        assert "API Key" in result["reason"]

    @pytest.mark.asyncio
    async def test_request_shape(self):
        client = FakeClient({"status": "ready", "answers": {}})
        await jev_judge.judge_via_jev(
            client,
            {"max_state_chars": 1200, "history_rounds": 2},
            self.PUZZLE,
            self.TRUTH,
            self.QUESTION,
            [{"question": "旧问题", "answer": "是"}],
        )
        call = client.calls[0]
        assert list(call["questions"]) == [jev_judge.QUESTION_NAME]
        assert call["questions"][jev_judge.QUESTION_NAME]["type"] == "choice"
        assert self.TRUTH in call["state"]
        assert self.QUESTION in call["state"]
