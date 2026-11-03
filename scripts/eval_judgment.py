#!/usr/bin/env python3
"""问答判定离线评测（需要真实的 Jev API Key）。

用途
----
1. **回归**：改了提示词（criteria / instructions / state 结构）之后，跑一遍固定用例，
   确认判定没有退化。提示词改动必须同时升级 ``core/jev_judge.PROMPT_VERSION``。
2. **验证上下文的作用**：同样的问题、带/不带历史问答各跑一遍，
   看最近几轮问答是否真的帮模型消解「他是不是…��这类指代。

用法
----
    export JEVMODEL_API_KEY=sk-xxxx
    python scripts/eval_judgment.py                 # 跑全部用例
    python scripts/eval_judgment.py --compare       # 额外对比「带/不带历史」
    python scripts/eval_judgment.py --base-url https://jevmodel.org

用例里的预期判定是人工标注的；判定准确率低于 --threshold（默认 0.9）时退出码非 0，
方便挂到 CI 上。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from astrbot_plugin_soupai.core import jev_judge  # noqa: E402
from astrbot_plugin_soupai.core.jev_transport import JevClient  # noqa: E402


class Case:
    """一个评测用例。

    ``expect`` 可以是单个判定，也可以是一个**可接受集合**——「说对一半」的
    提问在只有 是/否/无关 三档时本就没有唯一正解，模型选 是 或 否 都算对，
    区别会体现在置信度上。
    """

    def __init__(
        self,
        name: str,
        puzzle: str,
        truth: str,
        question: str,
        expect,
        recent: Optional[List[Dict[str, str]]] = None,
        note: str = "",
    ):
        self.name = name
        self.puzzle = puzzle
        self.truth = truth
        self.question = question
        self.expect = expect if isinstance(expect, (tuple, list)) else (expect,)
        self.recent = recent or []
        self.note = note

    def accepts(self, verdict: str) -> bool:
        return verdict in self.expect


# ── 用例（取自插件自带题库，人工标注预期）────────────────────────────────
CASES: List[Case] = [
    Case(
        name="灯塔管理员 / 明确否",
        puzzle="一个男人推开门，看到眼前的景象后立即跳楼自杀了。为什么？",
        truth="这个男人是灯塔管理员，他发现灯塔的灯灭了，意识到因为自己的疏忽导致船只失事，愧疚之下选择了跳楼。",
        question="这个死者是女性吗",
        expect="否",
        note="明确否",
    ),
    Case(
        name="灯塔管理员 / 明确是",
        puzzle="一个男人推开门，看到眼前的景象后立即跳楼自杀了。为什么？",
        truth="这个男人是灯塔管理员，他发现灯塔的灯灭了，意识到因为自己的疏忽导致船只失事，愧疚之下选择了跳楼。",
        question="他是因为愧疚才自杀的吗",
        expect="是",
        note="命中核心动机",
    ),
    Case(
        name="海龟汤 / 无关",
        puzzle="一个女人在餐厅点了一份海龟汤，喝了一口后就哭了。为什么？",
        truth="这个女人曾经和丈夫一起遇难，丈夫告诉她煮的是海龟汤让她活下来，但她现在才知道当时喝的其实是丈夫的肉做的汤。",
        question="餐厅的厨师今天穿的是蓝色制服吗",
        expect="不重要",
        note="与真相无关",
    ),
    Case(
        name="盲人与导盲犬 / 明确是",
        puzzle="一个人在深夜接到电话，听到几声后就挂断了，然后跳楼自杀。为什么？",
        truth="这个人是盲人，他养的导盲犬走失了。朋友打电话让他听狗叫声，想告诉他狗狗找到了，但这个人以为狗狗已经死了，绝望之下选择自杀。",
        question="他养的导盲犬走失了吗",
        expect="是",
        note="命中关键事实",
    ),
    Case(
        name="盲人与导盲犬 / 明确否",
        puzzle="一个人在深夜接到电话，听到几声后就挂断了，然后跳楼自杀。为什么？",
        truth="这个人是盲人，他养的导盲犬走失了。朋友打电话让他听狗叫声，想告诉他狗狗找到了，但这个人以为狗狗已经死了，绝望之下选择自杀。",
        question="他是因为被人杀害才死的吗",
        expect="否",
        note="死因是自杀",
    ),
    Case(
        name="指代消歧 / 带历史",
        puzzle="一个人在深夜接到电话，听到几声后就挂断了，然后跳楼自杀。为什么？",
        truth="这个人是盲人，他养的导盲犬走失了。朋友打电话让他听狗叫声，想告诉他狗狗找到了，但这个人以为狗狗已经死了，绝望之下选择自杀。",
        question="它还活着吗",
        expect="是",
        recent=[{"question": "他养的导盲犬走失了吗", "answer": "是"}],
        note="「它」= 导盲犬，只有带上历史才消得解",
    ),
    Case(
        name="说对一半 / 是或否皆可",
        puzzle="一个男人推开门，看到眼前的景象后立即跳楼自杀了。为什么？",
        truth="这个男人是灯塔管理员，他发现灯塔的灯灭了，意识到因为自己的疏忽导致船只失事，愧疚之下选择了跳楼。",
        question="他因为愧疚而自杀，但灯塔的灯其实早就坏了、并不是他的疏忽吗",
        expect=("是", "否"),
        note="前半句对（灯塔管理员、愧疚），后半句与真相冲突——三档下没有专属选项，"
             "答 是/否 都可，区别看置信度",
    ),
    Case(
        name="无关 / 真相提到过就别判无关",
        puzzle="一个男人推开门，看到眼前的景象后立即跳楼自杀了。为什么？",
        truth="这个男人是灯塔管理员，他发现灯塔的灯灭了，意识到因为自己的疏忽导致船只失事，愧疚之下选择了跳楼。",
        question="他是灯塔管理员吗",
        expect="是",
        note="真相顺带提到的事实也算数，不该判无关",
    ),
]


async def run_case(client: JevClient, cfg: Dict[str, Any], case: Case, use_history: bool):
    result = await jev_judge.judge_via_jev(
        client,
        cfg,
        case.puzzle,
        case.truth,
        case.question,
        case.recent if use_history else None,
        gate_mode=jev_judge.GATE_MODE_THRESHOLD,
        min_confidence=0.0,  # 评测时不过门控：先看模型怎么说，再看判定对不对
    )
    return result


async def main() -> int:
    parser = argparse.ArgumentParser(description="海龟汤问答判定离线评测")
    parser.add_argument("--base-url", default=os.environ.get("JEVMODEL_BASE_URL", "https://jevmodel.org"))
    parser.add_argument("--model", default=os.environ.get("JEVMODEL_MODEL", "jev-latest"))
    parser.add_argument("--api-key", default=os.environ.get("JEVMODEL_API_KEY", ""))
    parser.add_argument("--threshold", type=float, default=0.9, help="判定准确率下限")
    parser.add_argument("--compare", action="store_true", help="额外对比带/不带历史")
    parser.add_argument("--language", default=jev_judge.LANG_AUTO, choices=list(jev_judge.LANGUAGES))
    args = parser.parse_args()

    if not args.api_key:
        print("✗ 缺少 API Key：请设置环境变量 JEVMODEL_API_KEY")
        return 2

    cfg = {
        "api_key": args.api_key,
        "base_url": args.base_url,
        "model": args.model,
        "timeout_sec": 20,
        "max_state_chars": jev_judge.DEFAULT_MAX_STATE_CHARS,
        "history_rounds": jev_judge.DEFAULT_HISTORY_ROUNDS,
        "prompt_language": args.language,
    }

    client = JevClient()
    print("=" * 78)
    print(f"问答判定评测 · 模型 {args.model} · 提示词 {jev_judge.PROMPT_VERSION} · 语言 {args.language}")
    print("=" * 78)

    hits = 0
    total = 0
    try:
        for case in CASES:
            result = await run_case(client, cfg, case, use_history=True)
            verdict = result.get("verdict") or "-"
            confidence = result.get("confidence", 0.0)
            ok = case.accepts(verdict)
            hits += int(ok)
            total += 1
            mark = "✓" if ok else "✗"
            print(f"\n{mark} [{case.name}] {case.note}")
            print(f"   提问: {case.question}")
            expected = "/".join(case.expect)
            print(f"   预期: {expected}   实得: {verdict}   置信度: {confidence:.3f}")
            if not ok:
                print(f"   原因: {result.get('reason')}")

            if args.compare and case.recent:
                without = await run_case(client, cfg, case, use_history=False)
                print(
                    f"   对比(无历史): {without.get('verdict') or '-'} "
                    f"置信度 {without.get('confidence', 0.0):.3f}"
                )
    finally:
        await client.close()

    print()
    print("=" * 78)
    accuracy = hits / total if total else 0.0
    print(f"判定准确率: {hits}/{total} = {accuracy:.1%}（门槛 {args.threshold:.0%}）")
    if accuracy < args.threshold:
        print("✗ 未达门槛：先改题目或提示词，再重测受影响范围")
        return 1
    print("✓ 通过")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
