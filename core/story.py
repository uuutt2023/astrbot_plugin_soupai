"""自动生成谜题：调 LLM 写新汤，并从返回内容里解析出题面与答案。

解析要兼容好几种格式——模型爱加 Markdown、换行、冒号宽窄不一，
所以先试几种固定格式，再退回逐行扫描。
"""

from typing import Optional, Tuple

from astrbot.api import logger

from ..prompts import story as story_prompts


def parse_story(text: str) -> Tuple[Optional[str], Optional[str]]:
    """从 LLM 返回内容里解析出 (题面, 答案)。

    解析不出来时返回 (None, None)，由调用方给出失败文案。
    """
    puzzle = None
    answer = None

    # 格式1: "题面：xxx 答案：xxx"
    if "题面：" in text and "答案：" in text:
        puzzle = text.split("题面：")[1].split("答案：")[0].strip()
        answer = text.split("答案：")[1].strip()

    # 格式2: "**题面**：xxx **答案**：xxx" (Markdown格式)
    elif "**题面**" in text and "**答案**" in text:
        puzzle = text.split("**题面**")[1].split("**答案**")[0].strip()
        if puzzle.startswith("：") or puzzle.startswith(":"):
            puzzle = puzzle[1:].strip()
        answer = text.split("**答案**")[1].strip()
        if answer.startswith("：") or answer.startswith(":"):
            answer = answer[1:].strip()

    # 格式3: "题面：xxx\n答案：xxx"
    elif "题面：" in text and "\n答案：" in text:
        puzzle = text.split("题面：")[1].split("\n答案：")[0].strip()
        answer = text.split("\n答案：")[1].strip()

    # 格式4: 尝试从文本中提取题面和答案
    else:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # 寻找题面
            if not puzzle and ("题面" in line or "**题面**" in line):
                puzzle = line
                if "：" in line:
                    puzzle = line.split("：", 1)[1].strip()
                elif ":" in line:
                    puzzle = line.split(":", 1)[1].strip()
                # 移除可能的Markdown标记
                puzzle = puzzle.replace("**", "").replace("*", "").strip()

            # 寻找答案
            elif not answer and ("答案" in line or "**答案**" in line):
                answer = line
                if "：" in line:
                    answer = answer.split("：", 1)[1].strip()
                elif ":" in line:
                    answer = answer.split(":", 1)[1].strip()
                # 移除可能的Markdown标记
                answer = answer.replace("**", "").replace("*", "").strip()

            # 如果找到了题面但还没找到答案，继续寻找
            elif puzzle and not answer and len(line) > 20:
                # 可能是答案的开始
                answer = line

    if puzzle and answer:
        # 清理答案中的多余内容
        if "----" in answer:
            answer = answer.split("----")[0].strip()
        if "---" in answer:
            answer = answer.split("---")[0].strip()
        return puzzle, answer

    return None, None


async def generate_story(provider) -> Tuple[str, str]:
    """使用 LLM 生成海龟汤谜题，返回 (题面, 答案)。

    失败时返回 ("生成失败", 失败原因)——沿用原有的兜底文案。
    """
    if provider is None:
        logger.error("未配置 LLM 服务商")
        return "（无法生成题面，请先配置大语言模型）", "（无）"

    prompt = story_prompts.build_puzzle_prompt()

    try:
        logger.info("开始调用 LLM 生成谜题...")
        llm_resp = await provider.text_chat(
            prompt=prompt,
            contexts=[],
            func_tool=None,
            image_urls=[],
            system_prompt=story_prompts.STORY_SYSTEM_PROMPT,
        )

        text = llm_resp.completion_text.strip()
        logger.info(f"LLM 返回内容: {text}")

        puzzle, answer = parse_story(text)
        if puzzle and answer:
            logger.info(f"成功解析谜题: 题面='{puzzle}', 答案='{answer}'")
            return puzzle, answer

        logger.error(f"LLM 返回内容格式错误: {text}")
        return "生成失败", "无法解析 LLM 返回的内容"
    except Exception as e:
        logger.error(f"生成谜题失败: {e}")
        return "生成失败", f"LLM 调用出错: {e}"
