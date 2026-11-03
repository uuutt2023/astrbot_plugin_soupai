"""Jev 决策模型用的提示词：instructions + criteria。

与 ``prompts.judge``（LLM 后端的四档中文长提示词）是**两套不同的口径**：

- LLM 后端：是 / 否 / 是也不是 / 不重要 —— 让模型生成一句话再解析；
- Jev 后端：是 / 否 / 无关 —— 让模型在固定选项里选一个，带概率分布。

选项集对齐 Situation-Puzzle；判定程序参考 jev-turtle-soup 的做法写进
instructions。约束：instructions ≤ 1800 字符，criteria 序列化后 ≤ 2000 字符
（Jev 文档的上限，有测试守着）。
"""

# 中文 criteria：选项键就是玩家看到的那三个字，Jev 返回什么就显示什么。
# 每一档都写清「什么时候才用它」——「否」与「无关」最容易混。
CRITERIA_ZH = {
    "是": "玩家问的事实或合理推论被真相支持。",
    "否": (
        "真相明确排除了这个事实，答「是」会误导玩家时才用；"
        "真相只是没提到，不算「否」。"
    ),
    "无关": (
        "真相既没提到也没排除这个事实，或玩家问的根本不是关于这个故事的是非题。"
    ),
}

# 英文 criteria：与中文逐条对应
CRITERIA_EN = {
    "yes": "The claim or a reasonable inference from it is supported by the truth.",
    "no": (
        "The truth clearly rules the claim out, so that answering yes would "
        "mislead the player. The truth merely not mentioning it is not No."
    ),
    "irrelevant": (
        "The truth neither mentions nor rules out the fact, or the player did "
        "not ask a yes/no question about this story at all."
    ),
}

# 判定程序写进 instructions：先找事实、再定档，最后才考虑无关。
# 这一段是判定质量的关键（参考 jev-turtle-soup 的做法）。
INSTRUCTIONS_ZH = (
    "这是一个海龟汤（情境推理）游戏，你是主持人。玩家针对隐藏的真相提问或做出陈述，"
    "请判断该回哪一档。\n\n"
    "判定步骤：\n"
    "1. 先在「真相」里找玩家问的那个事实。真相里提到过它（哪怕只是顺带提到，"
    "或明确排除了它），就答「是」或「否」，不要答「无关」。\n"
    "2. 「否」只在真相明确排除了这个事实、答「是」会误导玩家时才用。\n"
    "3. 「无关」只在真相既没提到、也没排除这个事实时使用；"
    "或者玩家说的根本不是关于这个故事的是非题。\n"
    "4. 「历史问答」只用来消解「他」「那个」这类指代，绝不能当成证据。\n"
    "5. 玩家问的是整件事的解释（完整推理）而不是一个能核对的事实时，也答「无关」"
    "——那属于验证环节。\n\n"
    "例子：真相说他养的导盲犬走失了，接到朋友打来的电话。"
    "问「他养的狗走失了吗」＝是；问「他是被朋友杀害的吗」＝否；"
    "问「那天他在画画吗」（真相完全没提）＝无关。\n\n"
    "只看给出的真相、题面与历史问答，不要补充题面里没有的信息。"
)

INSTRUCTIONS_EN = (
    "This is a lateral-thinking puzzle game. You are the host: the player asks "
    "questions or makes statements about a hidden truth. Decide which single "
    "verdict their message deserves.\n\n"
    "How to decide:\n"
    "1. First look for the fact the player asks about in the truth. If the truth "
    "mentions it at all (even in passing, or explicitly rules it out), answer Yes "
    "or No - never Irrelevant.\n"
    "2. Use No only when the truth clearly rules the claim out, so that answering "
    "yes would mislead the player.\n"
    "3. Use Irrelevant only when the truth neither mentions nor rules out the "
    "fact, or the player did not ask a yes/no question about this story at all.\n"
    "4. The recent Q&A is only for resolving references such as 'he' or 'that'; "
    "it is never evidence.\n"
    "5. If the player asks for a full explanation instead of a checkable fact, "
    "answer Irrelevant - that belongs to the verification step.\n\n"
    "Example: the truth says his guide dog was lost and he received a call from a "
    "friend. 'Was his dog lost?' = Yes; 'Did his friend kill him?' = No; "
    "'Was he painting that day?' (never mentioned) = Irrelevant.\n\n"
    "Use only the truth, the puzzle surface and the recent Q&A; do not add "
    "information that is not in them."
)

# 英文模式下 state 的字段名
STATE_KEYS_EN = {
    "surface": "story",
    "answer": "answer",
    "history": "recentQAs",
    "question": "playerQuestion",
}

# 中文模式下 state 的字段名
STATE_KEYS_ZH = {
    "surface": "汤面",
    "answer": "真相",
    "history": "历史问答",
    "question": "玩家提问",
}
