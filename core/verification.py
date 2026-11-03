"""验证玩家推理：让 LLM 当裁判，给出四档等级与一句简评。

等级：
- 完全还原 / 核心推理正确 → 玩家猜中（``is_correct`` 为 True）
- 部分正确 / 基本不符 → 没猜中
"""

from astrbot.api import logger

from ..prompts import verification as verification_prompts
from .results import VerificationResult

# 算「猜中」的等级
CORRECT_LEVELS = ("完全还原", "核心推理正确")


def parse_verification_result(text: str) -> VerificationResult:
    """解析验证结果。格式不对时尝试从原文里兜底提取。"""
    try:
        # 提取等级和评价
        lines = text.strip().split("\n")
        level = ""
        comment = ""

        for line in lines:
            line = line.strip()
            if line.startswith("等级："):
                level = line.replace("等级：", "").strip()
            elif line.startswith("评价："):
                comment = line.replace("评价：", "").strip()

        # 判断是否猜中
        is_correct = level in list(CORRECT_LEVELS)

        if not level or not comment:
            # 如果解析失败，尝试从文本中提取信息
            if "完全还原" in text or "核心推理正确" in text:
                level = "核心推理正确" if "核心推理正确" in text else "完全还原"
                comment = "推理基本正确，但解析结果格式异常"
                is_correct = True
            else:
                level = "验证失败"
                comment = "无法解析验证结果"
                is_correct = False

        return VerificationResult(level, comment, is_correct)

    except Exception as e:
        logger.error(f"解析验证结果失败: {e}")
        return VerificationResult("验证失败", f"解析验证结果时发生错误: {e}")


async def verify_user_guess(provider, user_guess: str, true_answer: str) -> VerificationResult:
    """验证用户推理，返回 ``VerificationResult``。"""
    if provider is None:
        return VerificationResult("验证失败", "未配置 LLM，无法验证")

    system_prompt = verification_prompts.VERIFICATION_SYSTEM_PROMPT
    user_prompt = verification_prompts.build_verification_user_prompt(
        user_guess, true_answer
    )

    try:
        logger.info(f"开始验证用户推理: '{user_guess[:50]}'...")

        llm_resp = await provider.text_chat(
            prompt=user_prompt,
            contexts=[],
            func_tool=None,
            image_urls=[],
            system_prompt=system_prompt,
        )

        text = llm_resp.completion_text.strip()
        logger.info(f"验证 LLM 返回内容: {text}")

        return parse_verification_result(text)

    except Exception as e:
        logger.error(f"验证用户推理失败: {e}")
        return VerificationResult("验证失败", f"验证过程中发生错误: {e}")
