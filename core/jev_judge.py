"""海龟汤「玩家提问 → 四档判定」的 Jev 决策层。

这个模块是纯逻辑：只依赖标准库，不 import astrbot，因此测试不需要
AstrBot 桩件，直接跑 pytest 即可。

分工：
- 传输层（``core/jev_transport.py``）负责 HTTP、幂等、限额与错误码；
- 这里负责「问什么、怎么把答案翻成 是/否/是也不是/不重要、什么时候才采信」。

Jev 是决策模型：输入 state + 结构化问题，输出概率化的判定，不产自由文本。
玩家的提问本身就是一次四选一分类，所以直接用 Jev 的 choice 题型——
比让 LLM 生成一句话再解析它更省 token、也更稳。
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

# ── 四档判定 ────────────────────────────────────────────────────────────
# 与 main.py 原 LLM 提示词里的四个中文档一一对应，顺序也保持一致。
VERDICT_YES = "yes"
VERDICT_NO = "no"
VERDICT_PARTIAL = "partial"
VERDICT_IRRELEVANT = "irrelevant"

VERDICT_LABELS: Dict[str, str] = {
    VERDICT_YES: "是",
    VERDICT_NO: "否",
    VERDICT_PARTIAL: "是也不是",
    VERDICT_IRRELEVANT: "不重要",
}

# 问题名（标识符，只允许字母数字下划线）
QUESTION_NAME = "verdict"

DEFAULT_MAX_STATE_CHARS = 1200
DEFAULT_HISTORY_ROUNDS = 6
DEFAULT_MIN_CONFIDENCE = 0.6

# ── 门控口径 ────────────────────────────────────────────────────────────
GATE_MODE_THRESHOLD = "threshold"
GATE_MODE_SAMPLE = "sample"
GATE_MODES = (GATE_MODE_THRESHOLD, GATE_MODE_SAMPLE)

# ── Jev 请求体 ──────────────────────────────────────────────────────────
# instructions / criteria 一律英文：Jev 主训练语言是英文，中文描述的
# 判定精度明显更低。返回的 choice 键再映射回中文四档。
INSTRUCTIONS = (
    "A player is playing a lateral-thinking puzzle game and asks the judge a "
    "question or makes a statement about the hidden story. Decide which single "
    "verdict their message deserves. The player does not have to reveal the whole "
    "truth at once: a statement that is directionally correct and would not "
    "mislead the deduction deserves Yes rather than Yes-and-no; prefer Yes over "
    "Yes-and-no when only background information is missing; be careful with "
    "statements that could mislead the player and prefer Yes-and-no for those; "
    "balance the smoothness of the game against the difficulty of the deduction. "
    "Judge with the story context."
)

CRITERIA: Dict[str, str] = {
    VERDICT_YES: (
        "Yes - the player hits a key fact or action that matches the truth and "
        "helps the deduction move forward. Missing minor details can be ignored "
        "as long as the direction of the deduction stays correct."
    ),
    VERDICT_NO: (
        "No - the statement fully contradicts the truth, or contains an obvious "
        "error that would send the deduction in the wrong direction."
    ),
    VERDICT_PARTIAL: (
        "Yes and no - the player hits part of the truth, but the causal link is "
        "incomplete or biased, or the statement contains elements that could "
        "mislead the player, or it ignores a key point directly relevant to the "
        "story."
    ),
    VERDICT_IRRELEVANT: (
        "Irrelevant - the statement is unrelated to the truth of the story, or it "
        "cannot move the deduction forward at all."
    ),
}

# 传输层失败状态 → 一句可读原因（日志里要能一眼看懂为什么回退了）
_STATUS_REASON = {
    "unauthorized": "API Key 无效或已被禁用",
    "insufficient_credits": "Jev 账户余额不足",
    "conflict": "上一次请求仍在处理中",
    "invalid_request": "请求体未通过 Jev 校验",
    "invalid_answer": "Jev 返回的答案格式不合法",
    "rate_limited": "触发 Jev 限流",
    "upstream_error": "Jev 上游故障",
    "overloaded": "Jev 服务过载",
    "backoff": "Jev 正在退避等待",
    "busy": "Jev 并发已满",
    "unconfigured": "未配置 Jev API Key",
    "closed": "Jev 客户端已关闭",
    "error": "网络或响应异常",
}


def status_reason(status: str) -> str:
    """把传输层状态翻成一句人话。"""
    return _STATUS_REASON.get(str(status or ""), f"Jev 请求失败（{status or '未知'}）")


def build_questions() -> Dict[str, Dict[str, Any]]:
    """构造 Jev 请求的问题集合（单道 choice 题，落在文档的 8 道上限内）。"""
    return {
        QUESTION_NAME: {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": dict(CRITERIA),
        }
    }


def _history_lines(
    qa_history: Optional[List[Dict[str, str]]], history_rounds: int
) -> List[str]:
    """把历史问答压成 "Q: ...\\nA: ..." 的行，超出轮数只保留最近的。"""
    if not qa_history:
        return []
    rounds = max(0, int(history_rounds or 0))
    if rounds <= 0:
        return []
    lines: List[str] = []
    for item in list(qa_history)[-rounds:]:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "") or "").strip()
        answer = str(item.get("answer", "") or "").strip()
        if not question:
            continue
        lines.append(f"Q: {question}\nA: {answer}")
    return lines


def build_state(
    puzzle: str,
    truth: str,
    question: str,
    qa_history: Optional[List[Dict[str, str]]] = None,
    max_chars: int = DEFAULT_MAX_STATE_CHARS,
    history_rounds: int = DEFAULT_HISTORY_ROUNDS,
) -> str:
    """构造 state：题面 + 完整真相 + 最近几轮问答 + 玩家这句提问。

    超长时**从最旧的历史开始丢**——历史是锦上添花，真相和玩家这句才是
    判断的必需信息，宁可少带历史也不能截断真相或提问。
    """
    limit = max(200, int(max_chars or DEFAULT_MAX_STATE_CHARS))
    head = (
        "[Lateral-Thinking Puzzle] A player asks yes/no questions to deduce a "
        "hidden story. Judge the player's latest message.\n"
        f"[Scenario] {str(puzzle or '').strip()}\n"
        f"[Truth] {str(truth or '').strip()}"
    )
    tail = f"[Player] {str(question or '').strip()}"
    entries = _history_lines(qa_history, history_rounds)

    def assemble(history: List[str]) -> str:
        parts = [head]
        if history:
            parts.append("[History]\n" + "\n".join(history))
        parts.append(tail)
        return "\n".join(parts)

    state = assemble(entries)
    while len(state) > limit and entries:
        entries.pop(0)
        state = assemble(entries)
    if len(state) > limit:
        # 极端情况：题面/真相本身超长，硬截断（保留玩家提问在尾部）
        state = state[:limit]
    return state


def verdict_confidence(answer: Any) -> Tuple[Optional[str], Optional[float]]:
    """从 Jev 的 choice 答案里取出 (选中的档位, 置信度)。

    置信度优先取 top-1 的概率分布；没有分布时退回 ``confidence`` 字段。
    两者都缺失则返回 (档位, None)——调用方据此走「不采信」的保守路径。
    """
    if not isinstance(answer, dict):
        return None, None
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in VERDICT_LABELS:
        return None, None
    probabilities = answer.get("probabilities")
    confidence: Optional[float] = None
    if isinstance(probabilities, dict):
        top = probabilities.get(choice)
        if type(top) in (int, float):
            confidence = float(top)
    if confidence is None:
        raw = answer.get("confidence")
        if type(raw) in (int, float):
            confidence = float(raw)
    return choice, confidence


def decide_gate(
    probability: Optional[float],
    threshold: float = DEFAULT_MIN_CONFIDENCE,
    mode: str = GATE_MODE_THRESHOLD,
    rng: Optional[random.Random] = None,
) -> Tuple[bool, float]:
    """把「Jev 对自己这一档有多确定」变成「采不采信它」。

    两种口径：

    - ``threshold``：概率 ≥ 阈值才采信。行为可复现、成本可预期。
    - ``sample``：以概率 p 采信 Jev，否则回退 LLM。LLM 调用频率随 Jev 的
      不确定性线性上升——Jev 越没把握，越可能多花一次 LLM 的钱。

    Returns:
        tuple[bool, float]: (是否采信, 本次抽样值；threshold 口径恒为 0.0)
    """
    if probability is None:
        return False, 0.0
    if mode != GATE_MODE_SAMPLE:
        return probability >= float(threshold), 0.0
    draw = (rng or random).random()
    return draw <= probability, draw


def pick_verdict(
    answer: Any,
    threshold: float = DEFAULT_MIN_CONFIDENCE,
    mode: str = GATE_MODE_THRESHOLD,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """把 Jev 的 choice 答案翻成最终判定。

    Returns:
        dict: ``{"ok": bool, "parsed": bool, "verdict": str, "confidence": float,
        "accepted": bool, "draw": float, "reason": str}``
        ``parsed`` 表示答案结构合法（档位解出来了）；``accepted`` 表示门控通过；
        ``ok`` 是两者同时成立——只有 ok 为真时 ``verdict`` 才能拿去回复玩家。
    """
    choice, confidence = verdict_confidence(answer)
    if choice is None:
        return {
            "ok": False,
            "parsed": False,
            "verdict": "",
            "confidence": 0.0,
            "accepted": False,
            "draw": 0.0,
            "reason": "Jev 未返回合法档位",
        }
    accepted, draw = decide_gate(confidence, threshold, mode, rng)
    return {
        "ok": accepted,
        "parsed": True,
        "verdict": VERDICT_LABELS[choice],
        "choice": choice,
        "confidence": float(confidence or 0.0),
        "accepted": accepted,
        "draw": draw,
        "reason": "Jev 判定" if accepted else "Jev 置信度不足，回退 LLM",
    }


async def judge_via_jev(
    client,
    cfg: Dict[str, Any],
    puzzle: str,
    truth: str,
    question: str,
    qa_history: Optional[List[Dict[str, str]]] = None,
    gate_mode: str = GATE_MODE_THRESHOLD,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """跑一次 Jev 判断，返回可直接落地的判定结果。

    ``status`` 为 ``ready`` 时 ``verdict`` 是四档之一；其余情况都表示
    「Jev 这次没给出可用判定」，调用方应当回退 LLM——Jev 永远不能把游戏搞坏。
    """
    mode = str(gate_mode or GATE_MODE_THRESHOLD).strip() or GATE_MODE_THRESHOLD
    if mode not in GATE_MODES:
        mode = GATE_MODE_THRESHOLD
    try:
        threshold = float(min_confidence)
    except (TypeError, ValueError):
        threshold = DEFAULT_MIN_CONFIDENCE

    state = build_state(
        puzzle,
        truth,
        question,
        qa_history,
        max_chars=int(cfg.get("max_state_chars") or DEFAULT_MAX_STATE_CHARS),
        history_rounds=int(cfg.get("history_rounds") or DEFAULT_HISTORY_ROUNDS),
    )
    result = await client.evaluate(cfg, state, build_questions())
    status = str(result.get("status") or "")
    if status != "ready":
        return {
            "status": status or "error",
            "ok": False,
            "verdict": "",
            "confidence": 0.0,
            "reason": status_reason(status),
            "detail": str(result.get("error_message") or ""),
        }

    answers = result.get("answers") or {}
    answer = answers.get(QUESTION_NAME) if isinstance(answers, dict) else None
    picked = pick_verdict(answer, threshold, mode, rng)
    picked["status"] = "ready"
    picked["state_chars"] = len(state)
    picked["seconds"] = result.get("seconds")
    picked["usage"] = result.get("usage") or {}
    picked["model"] = result.get("model") or ""
    return picked
