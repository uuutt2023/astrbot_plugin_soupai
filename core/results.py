"""判定与验证的结果对象。

``JudgeResult`` 把「干净的判定」与「Jev 给的概率」分开存：
历史里只记干净的「是/否」，百分比只出现在给玩家的消息里。
"""

from typing import Dict


class VerificationResult:
    """验证结果类"""

    def __init__(self, level: str, comment: str, is_correct: bool = False):
        self.level = level
        self.comment = comment
        self.is_correct = is_correct

    def to_dict(self) -> Dict:
        return {
            "level": self.level,
            "comment": self.comment,
            "is_correct": self.is_correct,
        }


class JudgeResult:
    """一次问答判断的结果。

    - ``verdict``：干净的判定（是 / 否 / 是也不是 / 不重要），问答历史存这个；
    - ``confidence``：Jev 对该判定的把握（0 表示来自 LLM，LLM 不给概率）；
    - ``source``：判定来源（``llm`` / ``jev``）。

    拆开存是为了让「历史里的判定」保持干净，而「发给玩家的文本」可以按开关
    附带相关性百分比——百分比只是提示，不该污染历史和后续提示词。
    """

    def __init__(self, verdict: str, confidence: float = 0.0, source: str = "llm"):
        self.verdict = verdict
        self.confidence = confidence
        self.source = source

    def relevance_text(self) -> str:
        """相关性百分比文本，例如「相关性 88%」；LLM 判定或概率缺失时为空串。"""
        if self.source != "jev" or self.confidence <= 0:
            return ""
        percent = self.confidence * 100
        # 概率极小时四舍五入会变成 0%，那会误导人，直接写 <1%
        text = "<1%" if percent < 1 else f"{percent:.0f}%"
        return f"相关性 {text}"

    def display(self, count_text: str = "", show_relevance: bool = False) -> str:
        """最终发给玩家的文本。

        有提问计数时把计数与相关性放进同一组括号：「是（3/10，相关性 88%）」；
        没有计数时：「是（相关性 88%）」。
        """
        relevance = self.relevance_text() if show_relevance else ""
        if count_text:
            inner = count_text + (f"，{relevance}" if relevance else "")
            return f"{self.verdict}（{inner}）"
        if relevance:
            return f"{self.verdict}（{relevance}）"
        return self.verdict


# 自定义会话过滤器 - 以群为单位进行会话控制
