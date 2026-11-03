# soupai 接入 Jev 决策后端 —— 设计文档

日期：2026-11-03
仓库：https://github.com/uuutt2023/astrbot_plugin_soupai
分支：feature/jev-judge-backend

## 1. 背景与动机

海龟汤插件（`astrbot_plugin_soupai`）的每局游戏里，玩家每提一个问题，插件就要调用一次
LLM 去判断「玩家这句话该回 是 / 否 / 是也不是 / 不重要」。这是整个游戏最热的调用路径。

Jev 是一个决策模型服务：输入 `state` + 结构化问题（`noul` / `choice` / `score`），
返回概率化的判定，不产自由文本。它只对**输入 token** 计费，输出 token 免费，
且**上游失败不计费**（可安全重试）。这与本插件「判断」任务的需求高度吻合：
判断本质上就是一次四选一分类，不需要生成任何文字。

### 1.1 实测的性能画像（先测量再优化）

在离线 harness 中对纯 Python 路径做了测量（`/workspace/_probe/profile_soupai.py`）：

| 路径 | 实测中位耗时 | 频率 |
|---|---|---|
| `build_allow_list`（题面 + 20 轮问答） | 0.061 ms | 每次提示 |
| `NetworkSoupaiStorage.get_story`（含落盘） | 0.294 ms | 每次开局 |
| `LocalSoupaiStorage.add_story`（全量重写 JSON） | 0.582 ms | 每次自动生成 |
| 启动期 `network_soupai.json` 载入 | 3.97 ms | 每次启动 |

结论：**该插件的延迟几乎全部来自 LLM 调用（秒级），Python 侧没有值得动刀的热点。**
因此本次性能优化的主战场不是 CPU 微优化，而是把判断路径从「LLM 生成」换成
「Jev 判定」——延迟从 5~15s 降到 2~4s，成本从双向 token 降到只计输入 token。

## 2. 范围

### 2.1 纳入

- `judge_question`：玩家提问 → 四档判断。接入 Jev `choice` 题型。

### 2.2 不纳入（附理由）

- `generate_hint`（生成方向性提示）：需要产出一句自然语言提示，Jev 不产自由文本。
- `generate_story_with_llm`（自动生成谜题）：同上。
- `verify_user_guess`（验证玩家推理）：等级判定虽同样是四选一，但还要输出一句简评，
  而每局最多调用 2 次，价值低于复杂度。**列为后续扩展点**——等级判定可平移
  `core/jev_judge.py` 的 choice 机制。

## 3. 架构

```
main.py
 ├─ judge_question(question, true_answer, puzzle, qa_history)
 │    └─ 按 judge_backend 分发
 │         ├─ llm     → 原实现，行为完全不变
 │         ├─ jev     → JevClient.evaluate() → 四选一
 │         └─ hybrid  → Jev 先判，置信度不足 → 回落 llm
 │
 ├─ core/jev_judge.py   （纯逻辑，不 import astrbot，可独立单测）
 │    ├─ build_state()      题面 + 真相 + 最近 N 轮问答 + 玩家提问
 │    ├─ build_questions()  单道 choice 题，四档 criteria
 │    └─ decide_verdict()   门控（threshold / sample）+ 结果映射
 │
 └─ core/jev_transport.py （HTTP 层，幂等/限额/错误码/预算）
```

### 3.1 模块边界

- `core/jev_transport.py`：**通用**传输层，从 `astrbot_plugin_angel_heart` 复制而来
  （该版本已与 `astrbot_plugin_oven_multi` 对齐文档契约并通过实测）。仅两处差异：
  幂等键前缀 `angel-` → `soup-`（避免与其它插件在同一网关互撞键）；
  默认网关 `jevmodel`。
- `core/jev_judge.py`：**海龟汤专属**的判定逻辑，不 import astrbot，
  因此测试不需要 AstrBot 桩件。

### 3.2 为什么打破「单文件」惯例

仓库 `CLAUDE.md` 记录的实现形态是单文件插件。但 `main.py` 已达 2400 行，
再内联 500 行传输层会让它不可维护。AstrBot 按目录安装插件，额外模块可正常加载。
本次同步更新 `CLAUDE.md` 的架构章节。

## 4. Jev 请求契约

### 4.1 state（字符串）

```
[Lateral-Thinking Puzzle] A "Sea Turtle Soup" riddle game. Judge the player's message.
[Scenario] <题面>
[Truth] <完整真相>
[History] Q: ... A: ...        （最近 history_rounds 轮，超出 max_state_chars 从旧到新截断）
[Player] <玩家提问或陈述>
```

### 4.2 questions（单道 choice）

```json
{"verdict": {
  "type": "choice",
  "instructions": "Which single verdict does the player's message deserve? ...",
  "criteria": {
    "yes":        "Yes - ...",
    "no":         "No - ...",
    "partial":    "Yes and no - ...",
    "irrelevant": "Irrelevant - ..."
  }
}}
```

- 一次请求 1 道题（文档上限 8 道）。
- `criteria` 为 `{key: description}` 字典，`key` 即返回的 `answers.verdict.choice`。
- 返回：`{"type":"choice","choice":"yes","probabilities":{...},"confidence":0.93}`。
- 指令与选项描述一律英文（Jev 主训练语言为英文，CJK 精度低）；
  `choice` 键映射回中文四档：是 / 否 / 是也不是 / 不重要。

### 4.3 状态构造约束

- `max_state_chars` 默认 1200（与 angel_heart 一致），`history_rounds` 默认 6。
- 超限时**从旧到新丢弃**问答历史，优先保留真相与玩家提问。

## 5. 后端与门控

### 5.1 `judge_backend`（配置项，下拉）

| 值 | 行为 |
|---|---|
| `llm`（默认） | 完全走原实现。老用户零影响。 |
| `jev` | Jev 直出。Jev 未启用 / 无 Key / 超时 / 上游错误 → **自动回落 LLM**，游戏不中断。 |
| `hybrid` | Jev 先判；top-1 概率达标则采信，否则回落 LLM。 |

### 5.2 `gate_mode`（配置项，下拉）

| 值 | 行为 |
|---|---|
| `threshold`（默认） | top-1 概率 ≥ `min_confidence`（默认 0.6）才采信 Jev，否则回退 LLM。行为可复现、成本可预期。 |
| `sample` | 以 top-1 概率 p 采信 Jev，否则回退 LLM。LLM 调用频率随 Jev 不确定性线性上升，成本平滑。 |

抽样值写日志（3 位小数），事后可复盘当次为什么回落。

## 6. 错误处理

| 情况 | 行为 |
|---|---|
| Jev 未启用 / 未配 Key | 回落 LLM，日志提示 |
| HTTP 401 / 402 / 409 / 422 / 429 / 502 | 传输层按错误码处理；最终失败则回落 LLM |
| 超时 | 传输层重试；最终失败则回落 LLM |
| 响应结构非法（choice 不在 criteria 内等） | 传输层判为无效响应；最终失败则回落 LLM |
| LLM 也失败 | 保持原错误文案（`（判断失败，请重试）`） |

**回落原则：Jev 永远不能把游戏搞坏。** 所有 Jev 异常都在 `judge_question` 内兜住。

## 7. 配置项

新增分组 `jev_judge`：

| 键 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `enabled` | bool | false | 总开关 |
| `api_key` | string | "" | Jev API Key（sk- 开头） |
| `base_url` | string | `https://jevmodel.org` | 接口地址 |
| `model` | string | `jev-latest` | 模型名 |
| `timeout_sec` | float | 10.0 | 单次调用超时 |
| `retries` | int | 1 | 失败重试次数 |
| `gate_mode` | string | `threshold` | 门控口径 |
| `min_confidence` | float | 0.6 | 采信阈值 |
| `max_state_chars` | int | 1200 | state 长度上限 |
| `history_rounds` | int | 6 | 带入的历史问答轮数 |

顶层新增 `judge_backend`（下拉：llm / jev / hybrid）。

## 8. 测试

- `tests/test_jev_judge.py`：纯逻辑单测（不需要 AstrBot）
  - state 构造：字段齐全、历史轮数截断、长度上限
  - questions 形状：choice 四键、key 与映射表一致
  - 判定映射：四个 choice → 四个中文档
  - 门控：threshold 边界、sample 分布（固定随机源）、缺失概率
  - 回落：Jev 不可用 → LLM
- 离线 harness：构造真实响应 JSON，验证解析路径。
- 配置页：Chromium 实测下拉可选项与副标题渲染。

## 9. 性能工作清单

按「有测量支撑 + 不改语义」筛选：

| 项 | 状态 |
|---|---|
| 判断路径 LLM → Jev（延迟 5~15s → 2~4s；成本单向） | 核心 |
| `build_allow_list` 名单去重 list → set（保序） | 微优化，收益小，如实标注 |
| `save_usage_record` 事件循环内同步写盘 | 0.3ms，暂不动（改动风险大于收益） |
| 启动 JSON 载入 | 4ms，不动 |

## 10. 交付物

- `astrbot_plugin_soupai.zip`（插件整包）
- `soupai-jev-judge.patch`（相对上游基线的完整补丁）
- 配置页截图（`judge_backend` 下拉、`gate_mode` 下拉）
- 分支 `feature/jev-judge-backend` 推送到 `uuutt2023/astrbot_plugin_soupai`
