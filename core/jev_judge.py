"""海龟汤「玩家提问 → 四档判定」的 Jev 决策层。

这个模块是纯逻辑：只依赖标准库，不 import astrbot，因此测试不需要
AstrBot 桩件，直接跑 pytest 即可。

分工：
- 传输层（``core/jev_transport.py``）负责 HTTP、幂等、限额与错误码；
- 这里负责「问什么、怎么把答案翻成 是/否/无关、什么时候才采信」。

Jev 是决策模型：输入 state + 结构化问题，输出概率化的判定，不产自由文本。
玩家的提问本身就是一次四选一分类，所以直接用 Jev 的 choice 题型——
比让 LLM 生成一句话再解析它更省 token、也更稳。

设计参考 ``ZeroTang05/Situation-Puzzle``（生产级海龟汤 + Jev）的做法：

- **state 用结构化对象**：Jev 文档写明 state 可以是 string / object / array；
  用带字段名的对象（汤面 / 真相 / 历史问答 / 玩家提问）比拼接字符串更清晰，
  模型不必从「[Scenario] xxx」这类标记里猜字段边界。
- **提示词跟随题目语言**：题目是中文就用中文 criteria，选项标签与玩家
  看到的完全一致（Jev 直接返回「是」这样的标签，不用再做一层映射）；
  英文题目走英文 criteria。
- **缺概率算协议错误**：响应必须带合法选项与对应概率，缺概率不默认为 0，
  更不能拿一个来路不明的判定去回复玩家。
- **提示词有版本号**：改提示词要同时改版本号，便于事后追溯某次判定用的是
  哪套提示词。
- **判定程序写进 instructions**：「否」与「无关」最容易混，所以明确写死
  顺序：先在真相里找事实 → 找到就答 是/否（哪怕只是顺带提到）→ 真相明确
  排除才答 否 → 真相完全沉默才答 无关；历史问答只用于消解指代、不能当证据。
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from ..prompts import jev as jev_prompts

# 提示词版本：改了 instructions / criteria / state 结构就要升版本，
# 否则事后无法区分「判定变了」是提示词变了还是模型变了。
PROMPT_VERSION = "v2-2026-11-soupai"

# ── 三档判定（内部规范键，与语言无关）────────────────────────────────────
# 选项集对齐 Situation-Puzzle：是 / 否 / 无关。
# 原先还有一档「是也不是」，现在去掉了——那种「说对一半」的提问不再是第四个
# 选项，而是表现为**置信度不足**：分布摊在「是」和「否」之间，置信度门控会
# 把它拦下来交回 LLM 复核（见 decide_gate）。
VERDICT_YES = "yes"
VERDICT_NO = "no"
VERDICT_IRRELEVANT = "irrelevant"

# 玩家看到的标签
VERDICT_LABELS: Dict[str, str] = {
    VERDICT_YES: "是",
    VERDICT_NO: "否",
    VERDICT_IRRELEVANT: "无关",
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

# ── 语言 ────────────────────────────────────────────────────────────────
LANG_AUTO = "auto"
LANG_ZH = "zh"
LANG_EN = "en"
LANGUAGES = (LANG_AUTO, LANG_ZH, LANG_EN)

# criteria 的键 → 内部规范判定
CANONICAL: Dict[str, str] = {
    "是": VERDICT_YES,
    "否": VERDICT_NO,
    "无关": VERDICT_IRRELEVANT,
    "yes": VERDICT_YES,
    "no": VERDICT_NO,
    "irrelevant": VERDICT_IRRELEVANT,
}

# state 的字段名（按语言）——与提示词放在一起，改文案时不用翻两个文件
STATE_KEYS_ZH = jev_prompts.STATE_KEYS_ZH
STATE_KEYS_EN = jev_prompts.STATE_KEYS_EN

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


def detect_language(*texts: Any) -> str:
    """看文本里有没有中文，决定用哪套提示词。"""
    for text in texts:
        if not text:
            continue
        for ch in str(text):
            if "\u4e00" <= ch <= "\u9fff":
                return LANG_ZH
    return LANG_EN


def resolve_language(preferred: Any, *texts: Any) -> str:
    """决定本次判定用哪种语言：auto 时按内容自动判断。"""
    pref = str(preferred or LANG_AUTO).strip() or LANG_AUTO
    if pref not in LANGUAGES:
        pref = LANG_AUTO
    if pref == LANG_AUTO:
        return detect_language(*texts)
    return pref


def build_questions(language: str = LANG_ZH) -> Dict[str, Dict[str, Any]]:
    """构造 Jev 请求的问题集合（单道 choice 题，落在文档的 8 道上限内）。"""
    if language == LANG_EN:
        return {
            QUESTION_NAME: {
                "type": "choice",
                "instructions": jev_prompts.INSTRUCTIONS_EN,
                "criteria": dict(jev_prompts.CRITERIA_EN),
            }
        }
    return {
        QUESTION_NAME: {
            "type": "choice",
            "instructions": jev_prompts.INSTRUCTIONS_ZH,
            "criteria": dict(jev_prompts.CRITERIA_ZH),
        }
    }


def _history_lines(
    qa_history: Optional[List[Dict[str, str]]],
    history_rounds: int,
    language: str,
) -> List[str]:
    """把历史问答压成「问题 → 判定」。

    带判定标签而不是只带问题：模型能看出上一轮玩家问了什么、当时判了什么，
    这样「他是不是自缢」这类指代才有落点（参考项目评测里验证过：
    只在真歧义时改变结果，无回归）。
    """
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
        if language == LANG_EN:
            verdict = VERDICT_LABELS.get(answer, answer) if answer in CANONICAL else answer
            lines.append(f'"{question}" -> {verdict}')
        else:
            lines.append(f"「{question}」→ {answer}")
    return lines


def _state_length(state: Dict[str, Any]) -> int:
    """state 序列化后的长度（Jev 按序列化长度设限）。

    这里用 Python 默认分隔符（带空格）测量，比传输层实际发送的紧凑 JSON
    略长——宁可算多一点也不要在服务端被 422 拒掉。
    """
    try:
        return len(json.dumps(state, ensure_ascii=False))
    except (TypeError, ValueError):
        return 0


def build_state(
    puzzle: str,
    truth: str,
    question: str,
    qa_history: Optional[List[Dict[str, str]]] = None,
    max_chars: int = DEFAULT_MAX_STATE_CHARS,
    history_rounds: int = DEFAULT_HISTORY_ROUNDS,
    language: str = LANG_ZH,
) -> Dict[str, Any]:
    """构造结构化 state：汤面 / 真相 / 历史问答 / 玩家提问。

    超长时**从最旧的历史开始丢**——历史是锦上添花，真相和玩家这句才是
    判断的必需信息，宁可少带历史也不能截断真相或提问。
    """
    limit = max(200, int(max_chars or DEFAULT_MAX_STATE_CHARS))
    keys = STATE_KEYS_EN if language == LANG_EN else STATE_KEYS_ZH
    entries = _history_lines(qa_history, history_rounds, language)

    def assemble(history: List[str]) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            keys["surface"]: str(puzzle or "").strip(),
            keys["answer"]: str(truth or "").strip(),
        }
        if history:
            state[keys["history"]] = list(history)
        state[keys["question"]] = str(question or "").strip()
        return state

    state = assemble(entries)
    while _state_length(state) > limit and entries:
        entries.pop(0)
        state = assemble(entries)
    if _state_length(state) > limit:
        # 极端情况：题面/真相本身超长。按「真相 → 题面 → 提问」的顺序截断，
        # 真相优先保住（它承载判定所需的关键事实）。每次按超出量收缩，
        # 直到装得下或该字段已经见底。
        for field in (keys["answer"], keys["surface"], keys["question"]):
            while _state_length(state) > limit:
                value = state.get(field)
                if not isinstance(value, str) or len(value) <= 1:
                    break
                overflow = _state_length(state) - limit
                keep = max(0, len(value) - overflow - 1)
                state[field] = (value[:keep] + "…") if keep > 0 else ""
                if keep == 0:
                    break
    return state


def verdict_confidence(
    answer: Any, canonical: Optional[Dict[str, str]] = None
) -> Tuple[Optional[str], Optional[float]]:
    """从 Jev 的 choice 答案里取出 (内部规范判定, 置信度)。

    置信度优先取「被选中那一档的概率」；没有分布时退回 ``confidence`` 字段。
    两者都缺失则返回 (判定, None)——缺概率是协议错误，调用方不能拿它去回复玩家。
    """
    if not isinstance(answer, dict):
        return None, None
    choice = answer.get("choice")
    if not isinstance(choice, str):
        return None, None
    mapping = canonical or CANONICAL
    verdict = mapping.get(choice)
    if verdict is None:
        return None, None
    confidence: Optional[float] = None
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict):
        top = probabilities.get(choice)
        if type(top) in (int, float):
            confidence = float(top)
    if confidence is None:
        raw = answer.get("confidence")
        if type(raw) in (int, float):
            confidence = float(raw)
    return verdict, confidence


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
    canonical: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """把 Jev 的 choice 答案翻成最终判定。

    Returns:
        dict: ``{"ok": bool, "parsed": bool, "verdict": str, "confidence": float,
        "accepted": bool, "draw": float, "reason": str}``
        ``parsed`` 表示答案结构合法（判定解出来了）；``accepted`` 表示门控通过；
        ``ok`` 是两者同时成立——只有 ok 为真时 ``verdict`` 才能拿去回复玩家。
    """
    verdict, confidence = verdict_confidence(answer, canonical)
    if verdict is None:
        return {
            "ok": False,
            "parsed": False,
            "verdict": "",
            "confidence": 0.0,
            "accepted": False,
            "draw": 0.0,
            "reason": "Jev 返回的选项不在候选内（协议错误）",
        }
    if confidence is None:
        # 缺概率是协议错误，不能默认成 0——那等于拿一个来路不明的判定去回复玩家
        return {
            "ok": False,
            "parsed": True,
            "verdict": VERDICT_LABELS[verdict],
            "confidence": 0.0,
            "accepted": False,
            "draw": 0.0,
            "reason": "Jev 未返回该选项的概率（协议错误）",
        }
    accepted, draw = decide_gate(confidence, threshold, mode, rng)
    return {
        "ok": accepted,
        "parsed": True,
        "verdict": VERDICT_LABELS[verdict],
        "verdict_key": verdict,
        "confidence": float(confidence),
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

    language = resolve_language(
        cfg.get("prompt_language"), puzzle, truth, question
    )
    state = build_state(
        puzzle,
        truth,
        question,
        qa_history,
        max_chars=int(cfg.get("max_state_chars") or DEFAULT_MAX_STATE_CHARS),
        history_rounds=int(cfg.get("history_rounds") or DEFAULT_HISTORY_ROUNDS),
        language=language,
    )
    result = await client.evaluate(cfg, state, build_questions(language))
    status = str(result.get("status") or "")
    if status != "ready":
        return {
            "status": status or "error",
            "ok": False,
            "verdict": "",
            "confidence": 0.0,
            "reason": status_reason(status),
            "detail": str(result.get("error_message") or ""),
            "prompt_version": PROMPT_VERSION,
            "language": language,
        }

    answers = result.get("answers") or {}
    answer = answers.get(QUESTION_NAME) if isinstance(answers, dict) else None
    picked = pick_verdict(answer, threshold, mode, rng, CANONICAL)
    picked["status"] = "ready"
    picked["state_chars"] = _state_length(state)
    picked["seconds"] = result.get("seconds")
    picked["usage"] = result.get("usage") or {}
    picked["model"] = result.get("model") or ""
    picked["prompt_version"] = PROMPT_VERSION
    picked["language"] = language
    return picked
