"""问答判断：LLM 后端与 Jev 后端的统一编排。

两个后端回答的是同一个问题（玩家这句话该回哪一档），但口径不同：

- ``llm``：让 LLM 生成一句话，再解析出四档之一；
- ``jev``：让 Jev 决策模型在固定选项里选一个，带概率分布；
- ``hybrid``：Jev 先判，把握不足时回退 LLM。

三条路都不会把游戏搞坏：Jev 侧任何异常都回落到 LLM。
"""

from typing import Any, Dict, List, Optional

from astrbot.api import logger

from ..prompts import judge as judge_prompts
from . import jev_judge as jev
from .results import JudgeResult

# 后端标识
JUDGE_BACKEND_LLM = "llm"
JUDGE_BACKEND_JEV = "jev"
JUDGE_BACKEND_HYBRID = "hybrid"
JUDGE_BACKENDS = (JUDGE_BACKEND_LLM, JUDGE_BACKEND_JEV, JUDGE_BACKEND_HYBRID)

# 模型没好好回复时的兜底文案（沿用原样，玩家已经看惯了）
INVALID_REPLY = (
    "你给ai干宕机了或者有什么其他原因，反正他没好好回复，我也不知道为什么（我努力修过代码了）"
)

# 合法回答集合放模块级：每次判断都建一次 set 没必要
VALID_RESPONSES = set(judge_prompts.VALID_RESPONSES)


async def judge_with_llm(provider, question: str, true_answer: str) -> str:
    """用 LLM 判断玩家提问，返回四档之一；provider 为空时返回错误文案。"""
    if provider is None:
        return "（未配置 LLM，无法判断）"

    prompt = judge_prompts.build_judge_prompt(question, true_answer)

    try:
        llm_resp = await provider.text_chat(
            prompt=prompt,
            contexts=[],
            func_tool=None,
            image_urls=[],
            system_prompt=judge_prompts.JUDGE_SYSTEM_PROMPT,
        )

        valid_responses = VALID_RESPONSES
        reply = llm_resp.completion_text.strip()
        if reply in valid_responses:
            return reply
        return INVALID_REPLY

    except Exception as e:
        logger.error(f"判断问题失败: {e}")
        return "（判断失败，请重试）"


async def judge_with_jev(
    client,
    cfg: Dict[str, Any],
    puzzle: str,
    truth: str,
    question: str,
    qa_history: Optional[List[Dict[str, str]]] = None,
) -> Optional[JudgeResult]:
    """用 Jev 判断玩家提问；不可用时返回 None（调用方回退 LLM）。"""
    if not cfg.get("enabled"):
        logger.info("Jev 判断未启用，使用 LLM 判断")
        return None
    try:
        result = await jev.judge_via_jev(
            client,
            cfg,
            puzzle,
            truth,
            question,
            qa_history,
            gate_mode=cfg.get("gate_mode") or jev.GATE_MODE_THRESHOLD,
            min_confidence=cfg.get("min_confidence", jev.DEFAULT_MIN_CONFIDENCE),
        )
    except Exception as e:
        logger.error(f"Jev 判断异常，回退 LLM: {e}")
        return None

    if not result.get("ok"):
        logger.warning(
            f"Jev 未给出可用判定（{result.get('reason', '未知')}），回退 LLM"
        )
        return None
    logger.info(
        f"Jev 判定: {result.get('verdict')} "
        f"置信度 {result.get('confidence', 0.0):.3f} "
        f"提示词 {result.get('prompt_version', '')} "
        f"语言 {result.get('language', '')}"
    )
    return JudgeResult(
        str(result.get("verdict") or ""),
        confidence=float(result.get("confidence") or 0.0),
        source="jev",
    )


async def judge_question(
    provider_resolver,
    jev_client,
    backend: str,
    jev_cfg: Dict[str, Any],
    question: str,
    true_answer: str,
    puzzle: str = "",
    qa_history: Optional[List[Dict[str, str]]] = None,
) -> JudgeResult:
    """判断玩家提问，返回 ``JudgeResult``。

    ``provider_resolver`` 是个无参可调用，返回 ``(provider, 兜底文案)``：
    provider 为 None 时兜底文案就是给玩家看的那句「未配置 LLM，无法判断」。
    它是**惰性**调用的——Jev 判定成功时根本不需要 LLM，也不该打
    「未找到 LLM」的错误日志。

    Jev 后端拿不到可用判定时（未启用 / 超时 / 上游故障 / 返回结构不合法 /
    置信度不足）一律回退 LLM，玩家永远拿得到四档判定之一。
    """
    if backend in (JUDGE_BACKEND_JEV, JUDGE_BACKEND_HYBRID):
        judged = await judge_with_jev(
            jev_client, jev_cfg, puzzle, true_answer, question, qa_history
        )
        if judged is not None:
            return judged
    provider, fallback = provider_resolver()
    if provider is None:
        return JudgeResult(fallback or "（未配置 LLM，无法判断）", source="llm")
    return JudgeResult(
        await judge_with_llm(provider, question, true_answer), source="llm"
    )
