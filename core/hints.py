"""方向性提示：允许名词表 + 提示生成。

allow_list 是提示的护栏——提示里只能出现题面与历史问答里已经出现过的名词，
防止模型编造题面里没有的东西。
"""

import re
from typing import Dict, List

from astrbot.api import logger

from ..prompts import hint as hint_prompts

# 预编译：allow_list 每提取一个候选名词就要用一次，放在模块级避免
# 每次调用都走 re 的模式缓存查找，也免掉首次调用现场编译的卡顿。
_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")
_TRAILING_UPPER_RE = re.compile(r".*([A-Z])$")


def build_allow_list(puzzle: str, qa_history: List[Dict[str, str]]) -> List[str]:
    """根据题面和历史问答构建允许在提示中出现的名词列表。"""
    # 汇总文本：题面 + 所有问答
    parts = [puzzle] + [
        f"{item.get('question', '')}{item.get('answer', '')}" for item in qa_history
    ]
    text = "\n".join(parts)

    # 提取连续的中文、字母或数字片段作为候选名词
    tokens = _TOKEN_RE.findall(text)

    allow: List[str] = []
    seen = set()
    for token in tokens:
        if not token:
            continue
        # 同义合并：例如“男人A”“嫌疑人A”只保留末尾的大写字母
        m = _TRAILING_UPPER_RE.match(token)
        if m:
            token = m.group(1)
        if token not in seen:
            seen.add(token)
            allow.append(token)
    return allow


async def generate_hint(
    provider,
    puzzle: str,
    true_answer: str,
    qa_history: List[Dict[str, str]],
    hint_history: List[str],
    allow_list: List[str],
) -> str:
    """根据本局已记录的问答与提示生成新的方向性提示。"""
    if provider is None:
        return "（未配置 LLM，无法提供提示）"

    prompt = hint_prompts.build_hint_prompt(
        puzzle, true_answer, qa_history, hint_history, allow_list
    )

    try:
        llm_resp = await provider.text_chat(
            prompt=prompt,
            contexts=[],
            func_tool=None,
            image_urls=[],
        )
        text = llm_resp.completion_text.strip()
        if text.startswith("提示："):
            text = text[len("提示："):]
        return text
    except Exception as e:
        logger.error(f"生成提示失败: {e}")
        return "（生成提示失败，请重试）"
