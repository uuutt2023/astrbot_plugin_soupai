"""Jev 判断层（core/jev_judge.py）的纯逻辑测试。

这一层不 import astrbot，所以测试不需要框架环境。
"""

import importlib
import json
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
        questions = jev_judge.build_questions(jev_judge.LANG_ZH)
        assert list(questions) == [jev_judge.QUESTION_NAME]
        question = questions[jev_judge.QUESTION_NAME]
        assert question["type"] == "choice"
        assert question["instructions"].strip()
        assert len(question["criteria"]) == 3

    def test_chinese_criteria_keys_are_display_labels(self):
        """中文 criteria 的选项键就是玩家看到的那四个字，省掉一层映射。"""
        criteria = jev_judge.build_questions(jev_judge.LANG_ZH)[
            jev_judge.QUESTION_NAME
        ]["criteria"]
        assert set(criteria) == {"是", "否", "无关"}

    def test_english_criteria_keys(self):
        criteria = jev_judge.build_questions(jev_judge.LANG_EN)[
            jev_judge.QUESTION_NAME
        ]["criteria"]
        assert set(criteria) == {"yes", "no", "irrelevant"}

    def test_criteria_language_matches_request(self):
        zh = jev_judge.build_questions(jev_judge.LANG_ZH)[jev_judge.QUESTION_NAME]
        en = jev_judge.build_questions(jev_judge.LANG_EN)[jev_judge.QUESTION_NAME]
        assert any("一" <= ch <= "鿿" for ch in zh["instructions"])
        assert not any("一" <= ch <= "鿿" for ch in en["instructions"])
        assert any("一" <= ch <= "鿿" for ch in zh["criteria"]["是"])
        assert not any("一" <= ch <= "鿿" for ch in en["criteria"]["yes"])

    def test_instructions_within_documented_limit(self):
        for lang in (jev_judge.LANG_ZH, jev_judge.LANG_EN):
            instructions = jev_judge.build_questions(lang)[
                jev_judge.QUESTION_NAME
            ]["instructions"]
            assert len(instructions) <= 1800  # 文档上限

    def test_prompt_version_exists(self):
        assert jev_judge.PROMPT_VERSION


class TestLanguageDetection:
    def test_detect_chinese(self):
        assert jev_judge.detect_language("他是男性") == jev_judge.LANG_ZH

    def test_detect_english(self):
        assert jev_judge.detect_language("he is male") == jev_judge.LANG_EN

    def test_resolve_auto(self):
        assert jev_judge.resolve_language("auto", "他是男性") == jev_judge.LANG_ZH
        assert jev_judge.resolve_language("auto", "he is male") == jev_judge.LANG_EN

    def test_resolve_explicit_wins(self):
        assert jev_judge.resolve_language("zh", "he is male") == jev_judge.LANG_ZH
        assert jev_judge.resolve_language("en", "他是男性") == jev_judge.LANG_EN

    def test_invalid_preference_falls_back_to_auto(self):
        assert jev_judge.resolve_language("klingon", "他是男性") == jev_judge.LANG_ZH


# ── state 构造 ──────────────────────────────────────────────────────────
class TestBuildState:
    PUZZLE = "深夜停电，男主人死在上锁的浴室里。"
    TRUTH = "他滑倒撞到头后跌进浴缸溺水，锁是他自己装上去的。"
    QUESTION = "死者是不是滑倒的"

    def test_contains_all_parts(self):
        state = jev_judge.build_state(
            self.PUZZLE, self.TRUTH, self.QUESTION, [], language=jev_judge.LANG_ZH
        )
        assert isinstance(state, dict)
        assert state["汤面"] == self.PUZZLE
        assert state["真相"] == self.TRUTH
        assert state["玩家提问"] == self.QUESTION
        assert "历史问答" not in state

    def test_english_state_keys(self):
        state = jev_judge.build_state(
            self.PUZZLE, self.TRUTH, self.QUESTION, [], language=jev_judge.LANG_EN
        )
        assert state["story"] == self.PUZZLE
        assert state["answer"] == self.TRUTH
        assert state["playerQuestion"] == self.QUESTION

    def test_history_is_pairs_of_question_and_verdict(self):
        history = [{"question": "他是不是自缢", "answer": "否"}]
        state = jev_judge.build_state(
            self.PUZZLE, self.TRUTH, self.QUESTION, history, language=jev_judge.LANG_ZH
        )
        assert state["历史问答"] == ["「他是不是自缢」→ 否"]

    def test_history_rounds_limits_entries(self):
        history = [{"question": f"问题{i}", "answer": "是"} for i in range(10)]
        state = jev_judge.build_state(
            self.PUZZLE,
            self.TRUTH,
            self.QUESTION,
            history,
            history_rounds=3,
            language=jev_judge.LANG_ZH,
        )
        assert len(state["历史问答"]) == 3
        assert "问题0" not in state["历史问答"][0]
        assert "问题9" in state["历史问答"][-1]

    def test_history_rounds_zero_disables_history(self):
        history = [{"question": "问题0", "answer": "是"}]
        state = jev_judge.build_state(
            self.PUZZLE,
            self.TRUTH,
            self.QUESTION,
            history,
            history_rounds=0,
            language=jev_judge.LANG_ZH,
        )
        assert "历史问答" not in state

    def test_overlong_state_drops_oldest_history_first(self):
        history = [
            {"question": f"很长的历史问题{'x' * 200}{i}", "answer": "否"}
            for i in range(6)
        ]
        state = jev_judge.build_state(
            self.PUZZLE,
            self.TRUTH,
            self.QUESTION,
            history,
            max_chars=600,
            history_rounds=6,
            language=jev_judge.LANG_ZH,
        )
        assert len(json.dumps(state, ensure_ascii=False)) <= 600
        # 真相和玩家提问必须还在（判断的必需信息）
        assert state["真相"] == self.TRUTH
        assert state["玩家提问"] == self.QUESTION

    def test_extreme_overflow_is_truncated(self):
        state = jev_judge.build_state(
            "题面" * 500,
            "真相" * 500,
            "提问",
            None,
            max_chars=300,
            language=jev_judge.LANG_ZH,
        )
        assert len(json.dumps(state, ensure_ascii=False)) <= 300


# ── 答案解析 ────────────────────────────────────────────────────────────
class TestVerdictConfidence:
    def test_prefers_probability_of_selected_choice(self):
        answer = {
            "type": "choice",
            "choice": "否",
            "confidence": 0.9,
            "probabilities": {"是": 0.05, "否": 0.9, "是也不是": 0.03, "不重要": 0.02},
        }
        verdict, confidence = jev_judge.verdict_confidence(answer)
        assert verdict == jev_judge.VERDICT_NO
        assert confidence == pytest.approx(0.9)

    def test_falls_back_to_confidence_field(self):
        verdict, confidence = jev_judge.verdict_confidence(
            {"type": "choice", "choice": "是", "confidence": 0.42}
        )
        assert verdict == jev_judge.VERDICT_YES
        assert confidence == pytest.approx(0.42)

    def test_english_choice_is_mapped_to_canonical(self):
        verdict, confidence = jev_judge.verdict_confidence(
            {"type": "choice", "choice": "irrelevant", "probabilities": {"irrelevant": 0.7}}
        )
        assert verdict == jev_judge.VERDICT_IRRELEVANT
        assert confidence == pytest.approx(0.7)

    def test_unknown_choice_is_rejected(self):
        verdict, confidence = jev_judge.verdict_confidence(
            {"type": "choice", "choice": "maybe", "confidence": 0.9}
        )
        assert verdict is None
        assert confidence is None

    def test_non_dict_is_rejected(self):
        assert jev_judge.verdict_confidence(None) == (None, None)
        assert jev_judge.verdict_confidence("是") == (None, None)


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
            "probabilities": {
                k: (confidence if k == key else 0.0)
                for k in ("是", "否", "无关")
            },
        }

    @pytest.mark.parametrize(
        "key,label",
        [
            ("是", "是"),
            ("否", "否"),
            ("无关", "无关"),
        ],
    )
    def test_labels(self, key, label):
        picked = jev_judge.pick_verdict(self._answer(key, 0.95), 0.6, "threshold")
        assert picked["ok"] is True
        assert picked["accepted"] is True
        assert picked["verdict"] == label

    def test_low_confidence_falls_back(self):
        picked = jev_judge.pick_verdict(self._answer("是", 0.2), 0.6, "threshold")
        assert picked["ok"] is False
        assert picked["accepted"] is False
        assert picked["verdict"] == "是"  # 档位仍然解析出来了，只是没采信

    def test_invalid_answer_is_not_ok(self):
        picked = jev_judge.pick_verdict(
            {"type": "choice", "choice": "nope"}, 0.6, "threshold"
        )
        assert picked["ok"] is False
        assert picked["verdict"] == ""
        assert "协议错误" in picked["reason"]

    def test_missing_probability_is_protocol_error(self):
        """缺概率不能默认为 0——那等于拿一个来路不明的判定去回复玩家。"""
        picked = jev_judge.pick_verdict(
            {"type": "choice", "choice": "是"}, 0.6, "threshold"
        )
        assert picked["ok"] is False
        assert picked["parsed"] is True
        assert picked["confidence"] == 0.0
        assert "协议错误" in picked["reason"]


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
                        "choice": "否",
                        "confidence": 0.93,
                        "probabilities": {"是": 0.02, "否": 0.93},
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
        assert result["prompt_version"] == jev_judge.PROMPT_VERSION

    @pytest.mark.asyncio
    async def test_ready_but_unconfident_falls_back(self):
        client = FakeClient(
            {
                "status": "ready",
                "answers": {
                    "verdict": {
                        "type": "choice",
                        "choice": "是",
                        "confidence": 0.2,
                        "probabilities": {"是": 0.2},
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
        # state 是结构化对象（Jev 文档允许 string / object / array）
        assert isinstance(call["state"], dict)
        assert call["state"]["真相"] == self.TRUTH
        assert call["state"]["玩家提问"] == self.QUESTION
        assert call["state"]["历史问答"] == ["「旧问题」→ 是"]

    @pytest.mark.asyncio
    async def test_language_follows_content(self):
        client = FakeClient({"status": "ready", "answers": {}})
        await jev_judge.judge_via_jev(
            client, {}, "深夜的浴室", "他滑倒了", "他是不是滑倒的"
        )
        criteria = client.calls[0]["questions"][jev_judge.QUESTION_NAME]["criteria"]
        assert set(criteria) == {"是", "否", "无关"}

    @pytest.mark.asyncio
    async def test_language_can_be_forced(self):
        client = FakeClient({"status": "ready", "answers": {}})
        await jev_judge.judge_via_jev(
            client,
            {"prompt_language": "en"},
            "深夜的浴室",
            "他滑倒了",
            "他是不是滑倒的",
        )
        criteria = client.calls[0]["questions"][jev_judge.QUESTION_NAME]["criteria"]
        assert set(criteria) == {"yes", "no", "irrelevant"}


class TestDecisionProcedure:
    """判定程序必须写进 instructions，否则「否」和「无关」会混。"""

    def _instructions(self, lang):
        return jev_judge.build_questions(lang)[jev_judge.QUESTION_NAME]["instructions"]

    def test_zh_instructions_contain_key_rules(self):
        text = self._instructions(jev_judge.LANG_ZH)
        # 先在真相里找事实 → 找到就答是/否
        assert "先在「真相」里找" in text
        # 真相提到过就不要答无关
        assert "不要答「无关」" in text
        # 否只在明确排除时用
        assert "明确排除了这个事实" in text
        # 无关只在真相沉默时用
        assert "既没提到、也没排除" in text
        # 历史只用于消解指代，不能当证据
        assert "只用来消解" in text and "绝不能当成证据" in text

    def test_en_instructions_contain_key_rules(self):
        text = self._instructions(jev_judge.LANG_EN)
        assert "never Irrelevant" in text
        assert "clearly rules the claim out" in text
        assert "neither mentions nor rules out" in text
        assert "never evidence" in text

    def test_criteria_distinguish_no_from_irrelevant(self):
        criteria = jev_judge.build_questions(jev_judge.LANG_ZH)[
            jev_judge.QUESTION_NAME
        ]["criteria"]
        # 「否」必须写明「真相没提到不算否」
        assert "不算「否」" in criteria["否"]
        # 「无关」必须写明「既没提到也没排除」
        assert "既没提到也没排除" in criteria["无关"]

    def test_serialized_criteria_within_documented_limit(self):
        """文档限制：每道题的 criteria 序列化后不超过 2000 字符。

        判据越写越详细，越容易撞线——这条一旦破了就是必然的 422。
        """
        for lang in (jev_judge.LANG_ZH, jev_judge.LANG_EN):
            criteria = jev_judge.build_questions(lang)[
                jev_judge.QUESTION_NAME
            ]["criteria"]
            size = len(json.dumps(criteria, ensure_ascii=False))
            assert size <= 2000, f"{lang} 的 criteria 序列化后 {size} 字符，超上限"

    def test_instructions_within_documented_limit(self):
        for lang in (jev_judge.LANG_ZH, jev_judge.LANG_EN):
            instructions = jev_judge.build_questions(lang)[
                jev_judge.QUESTION_NAME
            ]["instructions"]
            assert len(instructions) <= 1800

    def test_only_three_options(self):
        """选项集对齐 Situation-Puzzle：是 / 否 / 无关。"""
        for lang in (jev_judge.LANG_ZH, jev_judge.LANG_EN):
            criteria = jev_judge.build_questions(lang)[
                jev_judge.QUESTION_NAME
            ]["criteria"]
            assert len(criteria) == 3
