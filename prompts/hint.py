"""方向性提示（hint）的提示词。

提示词要求模型只输出一句「关注【维度】：<下一步方向>」，并且只能使用
allow_list 里的名词，避免编造题面里没有的东西。
"""


def build_hint_prompt(
    puzzle: str,
    true_answer: str,
    qa_history,
    hint_history,
    allow_list,
) -> str:
    """构造方向性提示的用户提示词。"""
    history_text = "\n".join(
        [f"问：{item['question']}\n答：{item['answer']}" for item in qa_history]
    )
    hint_text = "\n".join(hint_history) if hint_history else "（无）"
    allow_text = ", ".join(allow_list) if allow_list else "（无）"
    return (
        "你是\"海龟汤\"提示生成器。你知道完整真相（仅供内部推理，严禁外泄）。\n"
        "材料：\n\n"
        f"* 题面：{puzzle}\n"
        f"* 完整真相（不可外泄）：{true_answer}\n"
        f"* 历史问答：{history_text}\n"
        f"* 历史提示：{hint_text}\n"
        f"* 允许名词 allow_list：{allow_text}（只能使用其中名词，不得创造新名词）\n\n"
        "在心中完成：\n\n"
        "1. 从历史问答归纳：已确认/已否定/不重要/部分正确的信息；\n"
        "2. 用以下维度整理：对象/身份、关系、动机、时间、地点、证据、步骤、先后、条件、规则、误解；\n"
        "3. 选择一个“未探索”或“partial 尚缺”的维度，且与历史提示不重复；\n"
        "4. 只使用 allow_list 中的名词与通用词，生成一句**动作化**的下一步提问方向；\n"
        "5. 禁止泄露真相细节，不得同义改写泄露；不得复述已确认内容。\n\n"
        "输出要求（只输出一句）：\n\n"
        "* 格式：关注【<维度>】：<动词 + allow_list名词/通用词>\n"
        "* 字数 ≤ 22（或 ≤ 24），不得添加解释。"
    )
