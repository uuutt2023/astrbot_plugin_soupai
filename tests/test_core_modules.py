"""core/ 与 prompts/ 里抽出来的纯函数的测试。

这些都是重构时从 main.py 搬出来的逻辑，用例锁住它们的行为，
以后再改这些模块时能第一时间发现行为漂移。
"""

import pytest

from astrbot_plugin_soupai.core import hints, story, verification
from astrbot_plugin_soupai.core.results import JudgeResult, VerificationResult
from astrbot_plugin_soupai.prompts import hint as hint_prompts
from astrbot_plugin_soupai.prompts import judge as judge_prompts
from astrbot_plugin_soupai.prompts import story as story_prompts
from astrbot_plugin_soupai.prompts import verification as ver_prompts


class TestParseStory:
    """造题返回内容的解析：兼容多种模型格式。"""

    def test_plain_format(self):
        puzzle, answer = story.parse_story("题面：男人死了\n答案：因为他喝了毒药")
        assert puzzle == "男人死了"
        assert answer == "因为他喝了毒药"

    def test_markdown_format(self):
        puzzle, answer = story.parse_story("**题面**：男人死了\n**答案**：他喝了毒药")
        assert puzzle == "男人死了"
        assert answer == "他喝了毒药"

    def test_trailing_horizontal_rule_trimmed(self):
        puzzle, answer = story.parse_story("题面：甲\n答案：乙\n----\n补充说明")
        assert (puzzle, answer) == ("甲", "乙")

    def test_unparsable_returns_none(self):
        assert story.parse_story("随便一段没有格式的回复") == (None, None)
        assert story.parse_story("题面：只有题面") == (None, None)


class TestParseVerificationResult:
    """验证结果的解析与「算不算猜中」。"""

    def test_correct_levels(self):
        for level in ("完全还原", "核心推理正确"):
            res = verification.parse_verification_result(f"等级：{level}\n评价：不错")
            assert res.level == level
            assert res.is_correct is True

    def test_wrong_levels(self):
        for level in ("部分正确", "基本不符"):
            res = verification.parse_verification_result(f"等级：{level}\n评价：偏差")
            assert res.is_correct is False

    def test_fallback_when_format_broken(self):
        res = verification.parse_verification_result("我觉得是核心推理正确")
        assert res.level == "核心推理正确"
        assert res.is_correct is True
        assert "格式异常" in res.comment

    def test_unparsable(self):
        res = verification.parse_verification_result("完全不知道")
        assert res.level == "验证失败"
        assert res.is_correct is False


class TestBuildAllowList:
    """提示允许名词表：只放题面与历史问答里出现过的名词。"""

    def test_extracts_tokens_and_dedupes(self):
        # 连续的中英文数字算一个 token（问答直接相接，所以"在房间"+"房间"是一个 token）
        allow = hints.build_allow_list(
            "一个男人 在房间", [{"question": "在房间", "answer": "房间"}]
        )
        assert allow == ["一个男人", "在房间", "在房间房间"]

    def test_trailing_upper_case_merged(self):
        # 以大写字母结尾的 token 只保留那个字母（"嫌疑人A" → "A"）
        allow = hints.build_allow_list("嫌疑人A", [])
        assert allow == ["A"]

    def test_empty_input(self):
        assert hints.build_allow_list("", [{"question": "", "answer": ""}]) == []


class TestJudgeResult:
    """判定结果：干净的判定与百分比分开存。"""

    def test_display_without_count(self):
        result = JudgeResult("否", confidence=0.88, source="jev")
        assert result.display("", True) == "否（相关性 88%）"

    def test_display_with_count(self):
        result = JudgeResult("否", confidence=0.88, source="jev")
        assert result.display("3/10", True) == "否（3/10，相关性 88%）"

    def test_display_without_relevance(self):
        result = JudgeResult("是", confidence=0.88, source="jev")
        assert result.display("3/10", False) == "是（3/10）"

    def test_llm_source_has_no_percentage(self):
        result = JudgeResult("是也不是", source="llm")
        assert "%" not in result.display("2/10", True)


class TestPromptBuilders:
    """提示词构造：内容必须真的带上要给的上下文。"""

    def test_judge_prompt_contains_truth_and_question(self):
        prompt = judge_prompts.build_judge_prompt("他在哪", "他在房间")
        assert "他在房间" in prompt
        assert "他在哪" in prompt

    def test_judge_prompt_lists_four_options(self):
        prompt = judge_prompts.build_judge_prompt("q", "a")
        for option in judge_prompts.VALID_RESPONSES:
            assert option in prompt

    def test_hint_prompt_contains_allow_list(self):
        prompt = hint_prompts.build_hint_prompt(
            "题面", "真相", [{"question": "q", "answer": "a"}], ["旧提示"], ["房间"]
        )
        assert "房间" in prompt
        assert "旧提示" in prompt

    def test_story_prompt_contains_one_theme(self):
        prompt = story_prompts.build_puzzle_prompt()
        assert any(theme in prompt for theme in story_prompts.THEMES)

    def test_verification_prompt_contains_both_sides(self):
        prompt = ver_prompts.build_verification_user_prompt("我认为是他杀", "其实是自杀")
        assert "我认为是他杀" in prompt
        assert "其实是自杀" in prompt


class TestVerificationResultObject:
    def test_to_dict(self):
        res = VerificationResult("完全还原", "很好", True)
        assert res.to_dict() == {
            "level": "完全还原",
            "comment": "很好",
            "is_correct": True,
        }


@pytest.mark.parametrize("name", ["judge", "hint", "story", "verification", "jev"])
def test_prompt_modules_are_importable(name):
    import importlib

    module = importlib.import_module(f"astrbot_plugin_soupai.prompts.{name}")
    assert module.__doc__
