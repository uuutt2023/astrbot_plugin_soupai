"""插件用到的全部提示词。

放在 ``prompts/`` 而不是业务代码里：改词不动逻辑，审阅时也能一眼看清
模型被要求做什么。模块划分按用途：

- ``judge``        LLM 后端的问答判断（四档中文长提示词）
- ``jev``          Jev 后端的 instructions + criteria（三档）
- ``hint``         方向性提示生成
- ``story``        自动生成谜题
- ``verification`` 验证玩家推理
"""

from . import hint, jev, judge, story, verification

__all__ = ["hint", "jev", "judge", "story", "verification"]
