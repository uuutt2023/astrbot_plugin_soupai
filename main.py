import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger, AstrBotConfig
from astrbot.core.utils.session_waiter import (
    session_waiter,
    SessionController,
    SessionFilter,
)
from astrbot.api.message_components import At

# 业务逻辑拆到 core/，提示词拆到 prompts/；这里只做接线
from .core import hints as hints_mod
from .core import jev_judge
from .core import judge as judge_mod
from .core import story as story_mod
from .core import verification as verification_mod
from .core.game_state import GameState
from .core.judge import (
    JUDGE_BACKEND_HYBRID,
    JUDGE_BACKEND_JEV,
    JUDGE_BACKEND_LLM,
    JUDGE_BACKENDS,
)
from .core.jev_transport import JevClient
from .core.results import JudgeResult, VerificationResult
from .core.storage import (
    CustomSoupaiStorage,
    LocalSoupaiStorage,
    NetworkSoupaiStorage,
    ThreadSafeStoryStorage,
)


# 自定义会话过滤器 - 以群为单位进行会话控制
class GroupSessionFilter(SessionFilter):
    """会话过滤器，确保每个群的会话独立"""

    def __init__(self, group_id: str):
        # 为每个会话保存其所属群 ID
        self.group_id = group_id

    def filter(self, event: AstrMessageEvent) -> str:
        current_group_id = (
            event.get_group_id() if event.get_group_id() else event.unified_msg_origin
        )
        # 仅当事件来自该群时才返回有效的会话 ID，否则返回空串避免误触发
        return self.group_id if current_group_id == self.group_id else ""


@register(
    "astrbot_plugin_soupai",
    "KONpiGG",
    "AI 海龟汤推理游戏插件，支持自动生成谜题、智能判断、验证系统、智能提示、存储库管理等功能。网络题库包含近300道海龟汤，还在持续更新中。",
    "1.0.3",
    "https://github.com/KONpiGG/astrbot_plugin_soupai",
)
class SoupaiPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.game_state = GameState()

        # 获取配置值
        self.generate_llm_provider_id = self.config.get("generate_llm_provider", "")
        self.judge_llm_provider_id = self.config.get("judge_llm_provider", "")
        self.game_timeout = self.config.get("game_timeout", 300)
        self.storage_max_size = self.config.get("storage_max_size", 50)
        self.auto_generate_start = self.config.get("auto_generate_start", 3)
        self.auto_generate_end = self.config.get("auto_generate_end", 6)
        self.puzzle_source_strategy = self.config.get(
            "puzzle_source_strategy", "network_first"
        )
        # TODO: 别名兼容处理，建议若干版本后删除
        if self.puzzle_source_strategy == "ai_first":
            self.puzzle_source_strategy = "local_first"

        # Jev 决策后端：判断问答走 LLM 还是走 Jev
        self.judge_backend = str(self.config.get("judge_backend", "llm") or "llm").strip()
        if self.judge_backend not in JUDGE_BACKENDS:
            self.judge_backend = JUDGE_BACKEND_LLM
        self.jev_judge_config = dict(self.config.get("jev_judge", {}) or {})
        self.jev_client = JevClient()


        # 难度设置
        self.difficulty_settings = {
            "简单": {
                "limit": None,
                "accept_levels": ["完全还原", "核心推理正确"],
                "hint_limit": 10,
            },
            "普通": {
                "limit": 35,
                "accept_levels": ["完全还原"],
                "hint_limit": 5,
            },
            "困难": {
                "limit": 15,
                "accept_levels": ["完全还原"],
                "hint_limit": 1,
            },
            "666开挂了": {
                "limit": 5,
                "accept_levels": ["完全还原"],
                "hint_limit": 0,
            },
        }
        self.group_difficulty: Dict[str, str] = {}

        # 数据存储路径: 使用框架提供的工具获取插件数据目录
        self.data_path = StarTools.get_data_dir()
        self.data_path.mkdir(parents=True, exist_ok=True)

        # 存储库初始化延迟到 init 方法中
        self.local_story_storage = None
        self.online_story_storage = None
        self.custom_story_storage = None

        # 防止重复调用的状态
        self.generating_games = set()  # 正在生成谜题的群聊ID集合

        # 自动生成状态
        self.auto_generating = False
        self.auto_generate_task = None

    @property
    def show_relevance(self) -> bool:
        """是否在判断结果后附带相关性百分比（Jev 判定时才有概率）。"""
        return bool(self.jev_judge_config.get("show_relevance", True))

    def _ensure_story_storages(self) -> None:
        """确保题库存储被初始化。

        在某些环境下, 插件的 ``init`` 方法可能未被调用或异常退出,
        导致存储对象仍为 ``None``。为避免后续调用出现
        ``'NoneType' object has no attribute 'get_story'`` 的错误, 这里
        提供一次性惰性初始化。
        """

        if self.local_story_storage is None:
            storage_file = self.data_path / "storage_soupai.json"
            self.local_story_storage = LocalSoupaiStorage(
                storage_file, self.storage_max_size, self.data_path
            )

        if self.online_story_storage is None:
            plugin_dir = Path(__file__).resolve().parent
            network_file = plugin_dir / "network_soupai.json"
            self.online_story_storage = NetworkSoupaiStorage(
                str(network_file), self.data_path
            )

        if self.custom_story_storage is None:
            custom_file = self.data_path / "custom_soupai.json"
            self.custom_story_storage = CustomSoupaiStorage(
                custom_file, self.data_path
            )

    async def init(self, context: Context):
        """插件初始化，此时 self.data_path 可用"""
        await super().init(context)

        # 初始化存储对象
        self._ensure_story_storages()

        # 启动自动生成任务
        asyncio.create_task(self._start_auto_generate())

        online_info = self.online_story_storage.get_storage_info()
        logger.info(
            f"海龟汤插件已加载，配置: 生成LLM提供商={self.generate_llm_provider_id}, 判断LLM提供商={self.judge_llm_provider_id}, 超时时间={self.game_timeout}秒, 网络题库={online_info['total']}个谜题, 本地存储库大小={self.storage_max_size}, 谜题来源策略={self.puzzle_source_strategy}"
        )

    async def terminate(self):
        """插件卸载时清理资源"""
        # 停止自动生成
        self.auto_generating = False
        if self.auto_generate_task:
            self.auto_generate_task.cancel()
        # 关闭 Jev 会话：否则 aiohttp 会留下未关闭的连接器
        try:
            await self.jev_client.close()
        except Exception as e:
            logger.error(f"关闭 Jev 客户端失败: {e}")
        logger.info("海龟汤插件已卸载呜呜呜呜呜")

    async def _start_auto_generate(self):
        """启动自动生成任务"""
        while True:
            try:
                now = datetime.now()
                current_hour = now.hour

                # 检查是否在自动生成时间范围内
                if self.auto_generate_start <= current_hour < self.auto_generate_end:
                    if not self.auto_generating:
                        # 检查存储库是否已满，如果已满则不启动自动生成
                        self._ensure_story_storages()
                        storage_info = self.local_story_storage.get_storage_info()
                        if storage_info["available"] <= 0:
                            logger.info(
                                f"本地存储库已满，跳过自动生成，时间: {current_hour}:00"
                            )
                            # 等待1小时后再次检查
                            await asyncio.sleep(3600)  # 1小时
                            continue

                        logger.info(f"开始自动生成故事，时间: {current_hour}:00")
                        self.auto_generating = True
                        asyncio.create_task(self._auto_generate_loop())
                else:
                    if self.auto_generating:
                        logger.info(f"停止自动生成故事，时间: {current_hour}:00")
                        self.auto_generating = False

                # 等待1小时后再次检查
                await asyncio.sleep(3600)  # 1小时
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"自动生成任务错误: {e}")
                await asyncio.sleep(3600)  # 出错后等待1小时再试

    async def _auto_generate_loop(self):
        """自动生成循环"""
        # 确保在运行循环前题库已初始化
        self._ensure_story_storages()
        while self.auto_generating:
            try:
                # 检查本地存储库是否已满
                storage_info = self.local_story_storage.get_storage_info()
                if storage_info["available"] <= 0:
                    logger.info("本地存储库已满，停止自动生成")
                    self.auto_generating = False
                    break

                # 生成一个故事
                puzzle, answer = await self.generate_story_with_llm()
                if puzzle and answer and not puzzle.startswith("（"):
                    self.local_story_storage.add_story(puzzle, answer)
                    logger.info("自动生成故事成功")
                else:
                    logger.warning("自动生成故事失败")

                # 等待5分钟再生成下一个
                await asyncio.sleep(300)  # 5分钟
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"自动生成故事错误: {e}")
                await asyncio.sleep(300)  # 出错后等待5分钟再试

    # ✅ 生成谜题和答案
    # ── AI 能力的薄封装 ──────────────────────────────────────────────
    # 具体实现在 core/ 下：judge=问答判断, hints=提示, story=造题,
    # verification=验证。这里只负责「按配置选 provider」和兜底文案。

    def _resolve_generate_provider(self):
        """解析生成谜题用的 LLM 提供商；找不到返回 None。"""
        if self.generate_llm_provider_id:
            provider = self.context.get_provider_by_id(self.generate_llm_provider_id)
            if provider is None:
                logger.error(
                    f"未找到指定的生成 LLM 提供商: {self.generate_llm_provider_id}"
                )
            return provider
        return self.context.get_using_provider()

    def _resolve_judge_provider(self):
        """解析判断/提示/验证用的 LLM 提供商；找不到返回 None。"""
        if self.judge_llm_provider_id:
            provider = self.context.get_provider_by_id(self.judge_llm_provider_id)
            if provider is None:
                logger.error(
                    f"未找到指定的判断 LLM 提供商: {self.judge_llm_provider_id}"
                )
            return provider
        return self.context.get_using_provider()

    def _judge_provider_or_message(self):
        """惰性取判断用的 LLM 提供商，返回 (provider, 兜底文案)。

        惰性是有意的：Jev 判定成功时，不该再打「未找到 LLM」的错误日志。
        """
        if self.judge_llm_provider_id:
            provider = self.context.get_provider_by_id(self.judge_llm_provider_id)
            if provider is None:
                logger.error(
                    f"未找到指定的判断 LLM 提供商: {self.judge_llm_provider_id}"
                )
                return None, "（未配置判断 LLM，无法判断）"
            return provider, None
        provider = self.context.get_using_provider()
        if provider is None:
            return None, "（未配置 LLM，无法判断）"
        return provider, None

    async def generate_story_with_llm(self) -> Tuple[str, str]:
        """使用 LLM 生成海龟汤谜题（实现见 core.story）"""
        provider = self._resolve_generate_provider()
        if provider is None:
            if self.generate_llm_provider_id:
                return "（无法生成题面，指定的生成 LLM 提供商不存在）", "（无）"
            logger.error("未配置 LLM 服务商")
            return "（无法生成题面，请先配置大语言模型）", "（无）"
        return await story_mod.generate_story(provider)

    async def verify_user_guess(
            self, user_guess: str, true_answer: str
    ) -> VerificationResult:
        """验证用户推理（实现见 core.verification）"""
        provider = self._resolve_judge_provider()
        if provider is None:
            if self.judge_llm_provider_id:
                return VerificationResult("验证失败", "未配置判断 LLM，无法验证")
            return VerificationResult("验证失败", "未配置 LLM，无法验证")
        return await verification_mod.verify_user_guess(
            provider, user_guess, true_answer
        )

    async def judge_question(
        self,
        question: str,
        true_answer: str,
        puzzle: str = "",
        qa_history: Optional[List[Dict[str, str]]] = None,
    ) -> JudgeResult:
        """判断玩家提问的回答方式：是 / 否 / 是也不是 / 不重要。

        按 ``judge_backend`` 分发：

        - ``llm``：走原来的 LLM 判断，行为与接入 Jev 之前完全一致；
        - ``jev``：走 Jev 决策模型（只计输入 token，不需要 LLM 生成文字）；
        - ``hybrid``：Jev 先判，置信度不够时再回退 LLM。

        Jev 永远不能把游戏搞坏：Jev 侧任何异常都会回落到 LLM。
        """
        return await judge_mod.judge_question(
            self._judge_provider_or_message,
            self.jev_client,
            self.judge_backend,
            self.jev_judge_config,
            question,
            true_answer,
            puzzle,
            qa_history,
        )

    def build_allow_list(self, puzzle: str, qa_history: List[Dict[str, str]]) -> List[str]:
        """构建提示里允许出现的名词表（实现见 core.hints）"""
        return hints_mod.build_allow_list(puzzle, qa_history)

    async def generate_hint(
            self,
            puzzle: str,
            true_answer: str,
            qa_history: List[Dict[str, str]],
            hint_history: List[str],
            allow_list: List[str],
    ) -> str:
        """根据本局已记录的问答与提示生成新的方向性提示（实现见 core.hints）"""
        provider = self._resolve_judge_provider()
        if provider is None:
            if self.judge_llm_provider_id:
                return "（未配置判断 LLM，无法提供提示）"
            return "（未配置 LLM，无法提供提示）"
        return await hints_mod.generate_hint(
            provider, puzzle, true_answer, qa_history, hint_history, allow_list
        )

    async def _generate_for_storage(self) -> bool:
        """为存储库生成故事"""
        try:
            puzzle, answer = await self.generate_story_with_llm()
            if puzzle and answer and not puzzle.startswith("（"):
                self.local_story_storage.add_story(puzzle, answer)
                logger.info("为存储库生成故事成功")
                return True
            else:
                logger.warning("为存储库生成故事失败")
                return False
        except Exception as e:
            logger.error(f"为存储库生成故事错误: {e}")
            return False

    # ✅ 验证用户推理
    # ✅ 判断提问的回答方式
    # ✅ 生成方向性提示
    @filter.command("汤难度")
    async def set_difficulty(self, event: AstrMessageEvent, level: str = ""):
        """设置游戏难度"""
        group_id = event.get_group_id()
        if not group_id:
            yield event.plain_result("此功能只能在群聊中使用")
            return
        if self.game_state.is_game_active(group_id):
            yield event.plain_result("当前有活跃游戏，无法修改难度")
            return
        if level not in self.difficulty_settings:
            options = "/".join(self.difficulty_settings.keys())
            current = self.group_difficulty.get(group_id, "普通")
            yield event.plain_result(f"可选难度：{options}\n当前难度：{current}")
            return
        self.group_difficulty[group_id] = level
        yield event.plain_result(f"难度已设置为 {level}")

    # 🎮 开始游戏指令
    @filter.command("汤")
    async def start_soupai_game(self, event: AstrMessageEvent):
        """开始海龟汤游戏
        
        使用格式: /汤 [题库类型] [题号]
        
        参数说明:
        - 题库类型 (可选): network(网络题库), storage(本地存储库), custom(自定义题库)
        - 题号 (可选): 指定题库中的题目索引，从0开始
        
        示例:
        /汤                    # 使用配置的策略随机获取谜题
        /汤 network           # 从网络题库随机获取谜题
        /汤 storage 5         # 从本地存储库获取第5号谜题
        /汤 custom 2          # 从自定义题库获取第2号谜题
        """
        group_id = event.get_group_id()
        logger.info(f"收到开始游戏指令，群ID: {group_id}")

        if not group_id:
            yield event.plain_result("海龟汤游戏只能在群聊中进行哦~")
            return

        # 检查是否已有活跃游戏
        if self.game_state.is_game_active(group_id):
            logger.info(f"群 {group_id} 已有活跃游戏")
            yield event.plain_result(
                "当前群聊已有活跃的海龟汤游戏，请等待游戏结束或使用 /揭晓 结束当前游戏。"
            )
            return

        # 检查是否正在生成谜题
        if group_id in self.generating_games:
            logger.info(f"群 {group_id} 正在生成谜题，忽略重复请求")
            yield event.plain_result("当前有正在生成的谜题，请稍候...")
            return

        self._ensure_story_storages()

        try:
            # 标记正在生成谜题
            self.generating_games.add(group_id)
            logger.info(f"开始为群 {group_id} 生成谜题")

            # 解析命令参数
            message_content = event.message_str.strip()
            args = message_content.split()[1:]  # 去掉命令本身
            
            story = None
            source_type = None
            puzzle_index = None
            
            # 解析参数格式: /汤 <network|storage|custom> <题号>
            # 两个参数都是可选的
            if len(args) >= 1:
                first_arg = args[0].lower()
                
                # 检查第一个参数是否是题库类型
                if first_arg in ["network", "local", "custom"]:
                    source_type = first_arg
                    
                    # 检查是否有第二个参数（题号）
                    if len(args) >= 2:
                        try:
                            puzzle_index = int(args[1])
                        except ValueError:
                            yield event.plain_result("题号必须是数字")
                            self.generating_games.discard(group_id)
                            return
                else:
                    # 第一个参数不是题库类型，可能是题号
                    try:
                        puzzle_index = int(first_arg)
                        # 使用配置的策略作为默认题库类型
                        strategy = self.puzzle_source_strategy
                        if strategy == "network_first":
                            source_type = "network"
                        elif strategy == "local_first":
                            source_type = "local"
                        elif strategy == "custom_first":
                            source_type = "custom"
                        else:  # random
                            source_type = "current"
                    except ValueError:
                        # 第一个参数既不是题库类型也不是题号，使用默认策略随机获取
                        source_type = "current"
            else:
                # 没有参数，使用配置的策略随机获取
                source_type = "current"
            
            # 根据解析的参数获取故事
            if puzzle_index is not None:
                # 指定了题号，从特定题库获取
                story = await self.get_story_by_index(source_type, puzzle_index)
                if not story:
                    yield event.plain_result(f"{source_type}题库中没有第 {puzzle_index} 号题目")
                    self.generating_games.discard(group_id)
                    return
            else:
                # 没有指定题号，根据策略随机获取
                if source_type == "current":
                    # 使用配置的策略获取随机故事
                    strategy = self.puzzle_source_strategy
                    story = await self.get_story_by_strategy(strategy)
                else:
                    # 从指定题库获取随机故事
                    if source_type == "network":
                        story = self.online_story_storage.get_story()
                    elif source_type == "local":
                        story = self.local_story_storage.get_story()
                    elif source_type == "custom":
                        story = self.custom_story_storage.get_story()
                    else:
                        yield event.plain_result("题库类型参数错误，请使用 network/local/custom")
                        self.generating_games.discard(group_id)
                        return


            if not story:
                yield event.plain_result("获取谜题失败，请重试")
                self.generating_games.discard(group_id)
                return

            puzzle, answer = story

            # 检查LLM生成是否失败
            if puzzle == "（无法生成题面，请先配置大语言模型）":
                yield event.plain_result(f"生成谜题失败：{answer}")
                self.generating_games.discard(group_id)
                return


            difficulty = self.group_difficulty.get(group_id, "普通")
            diff_conf = self.difficulty_settings.get(
                difficulty, self.difficulty_settings["普通"]
            )

            if self.game_state.start_game(
                    group_id,
                    puzzle,
                    answer,
                    difficulty=difficulty,
                    question_limit=diff_conf["limit"],
                    question_count=0,
                    verification_attempts=0,
                    accept_levels=diff_conf["accept_levels"],
                    hint_limit=diff_conf.get("hint_limit"),
                    hint_count=0,
            ):
                extra = ""
                if diff_conf["limit"] is not None:
                    extra = f"\n模式：{difficulty}（{diff_conf['limit']} 次提问"
                else:
                    extra = f"\n模式：{difficulty}（无限提问"

                hint_limit = diff_conf.get("hint_limit")
                if hint_limit == 0:
                    extra += "，无提示）"
                elif hint_limit is not None:
                    extra += f"，{hint_limit} 次提示）"
                else:
                    extra += "）"

                yield event.plain_result(
                    f"🎮 海龟汤游戏开始！{extra}\n\n📖 题面：{puzzle}\n\n💡 请直接提问或陈述，我会回答：是、否、是也不是\n💡 输入 /提示 可以获取方向性提示\n💡 输入 /验证 <答案> 可以验证答案是否正确\n💡 输入 /揭晓 可以查看完整故事"
                )

                # 启动会话控制
                await self._start_game_session(event, group_id, answer)
            else:
                yield event.plain_result("游戏启动失败，请重试")

            # 移除生成状态，因为故事已经准备完成
            self.generating_games.discard(group_id)
            logger.info(f"群 {group_id} 故事准备完成，移除生成状态")

        except Exception as e:
            logger.error(f"启动游戏失败: {e}")
            # 发生异常时也要移除生成状态
            self.generating_games.discard(group_id)
            logger.info(f"群 {group_id} 启动游戏异常，移除生成状态")
            yield event.plain_result(f"启动游戏时发生错误：{e}")

    # 🔍 揭晓指令
    @filter.command("揭晓")
    async def reveal_answer(self, event: AstrMessageEvent):
        """揭晓答案"""
        group_id = event.get_group_id()

        if not group_id:
            yield event.plain_result("海龟汤游戏只能在群聊中进行哦~")
            return

        # 检查是否有活跃游戏，如果有活跃游戏，说明在会话控制中，不在这里处理
        if self.game_state.is_game_active(group_id):
            # 阻止事件继续传播，避免被会话控制系统重复处理
            await event.block()
            return
        game = self.game_state.get_game(group_id)
        if not game:
            yield event.plain_result(
                "当前没有活跃的海龟汤游戏，请使用 /汤 开始新游戏。"
            )
            return

        answer = game["answer"]
        puzzle = game["puzzle"]

        # 发送完整的揭晓信息
        yield event.plain_result(
            f"🎯 海龟汤游戏结束！\n\n📖 题面：{puzzle}\n📖 完整故事：{answer}\n\n感谢参与游戏！"
        )

        # 结束游戏
        self.game_state.end_game(group_id)
        logger.info(f"游戏已结束，群ID: {group_id}")

    # 🎯 游戏会话控制
    async def _start_game_session(
            self, event: AstrMessageEvent, group_id: str, answer: str
    ):
        """启动游戏会话控制"""
        try:

            @session_waiter(timeout=self.game_timeout, record_history_chains=False)
            async def game_session_waiter(
                    controller: SessionController, event: AstrMessageEvent
            ):
                try:
                    # 从游戏状态获取答案，确保变量可用
                    game = self.game_state.get_game(group_id)
                    if not game:
                        return
                    current_answer = game["answer"]
                    user_input = event.message_str.strip()
                    logger.info(f"会话控制收到消息: '{user_input}'")

                    # 允许在会话中使用 /汤状态 和 /强制结束 指令
                    if user_input in ("/汤状态", "汤状态"):
                        await self._handle_game_status_in_session(event, group_id)
                        return

                    if user_input in ("/强制结束", "强制结束"):
                        await self._handle_force_end_in_session(event, group_id)
                        if not self.game_state.is_game_active(group_id):
                            controller.stop()
                        return

                    normalized_input = user_input.lstrip("/").strip()
                    if normalized_input == "查看":
                        await self._handle_view_history_in_session(event, group_id)
                        controller.keep(timeout=self.game_timeout, reset_timeout=True)
                        return
                    if user_input in ("/提示", "提示"):

                        async for result in self.hint_command(event):
                            await event.send(result)
                        controller.keep(timeout=self.game_timeout, reset_timeout=True)
                        return
                    # 特殊处理 /验证 指令
                    if user_input.startswith("/验证"):
                        import re

                        match = re.match(r"^/验证\s*(.+)$", user_input)
                        if match:
                            user_guess = match.group(1).strip()
                            # 手动调用验证函数
                            await self._handle_verification_in_session(
                                event, user_guess, current_answer
                            )
                            # 检查游戏是否已结束（用户可能猜中了）
                            if not self.game_state.is_game_active(group_id):
                                controller.stop()
                                return
                        else:
                            await event.send(
                                event.plain_result(
                                    "请输入要验证的内容，例如：/验证 他是她的父亲"
                                )
                            )
                        return
                    elif user_input.startswith("验证"):
                        import re

                        match = re.match(r"^验证\s*(.+)$", user_input)
                        if match:
                            user_guess = match.group(1).strip()
                            # 手动调用验证函数
                            await self._handle_verification_in_session(
                                event, user_guess, current_answer
                            )
                            # 检查游戏是否已结束（用户可能猜中了）
                            if not self.game_state.is_game_active(group_id):
                                controller.stop()
                                return
                        else:
                            await event.send(
                                event.plain_result(
                                    "请输入要验证的内容，例如：验证 他是她的父亲"
                                )
                            )
                        return
                    # 特殊处理 /揭晓 指令
                    if user_input == "揭晓":
                        # 获取游戏信息并发送答案
                        game = self.game_state.get_game(group_id)
                        if game:
                            answer = game["answer"]
                            puzzle = game["puzzle"]
                            await event.send(
                                event.plain_result(
                                    f"🎯 海龟汤游戏结束！\n\n📖 题面：{puzzle}\n📖 完整故事：{answer}\n\n感谢参与游戏！"
                                )
                            )
                            self.game_state.end_game(group_id)
                        controller.stop()
                        return
                    # Step 1: 检查是否是 /开头的命令，如果是则忽略，让指令处理器处理
                    if user_input.startswith("/"):
                        # 不处理指令，让事件继续传播到指令处理器
                        return
                    # Step 2: 检查是否 @了 bot，只有@bot的消息才触发问答判断
                    if not self._is_at_bot(event):
                        return
                    # Step 3: 是@bot的自然语言提问，触发 LLM 判断
                    game = self.game_state.get_game(group_id)
                    question_limit = game.get("question_limit") if game else None
                    question_count = game.get("question_count", 0) if game else 0
                    if question_limit is not None and question_count >= question_limit:
                        remaining = 2 - game.get("verification_attempts", 0)
                        await event.send(
                            event.plain_result(
                                f"❗️提问次数已用完，请使用 /验证 进行猜测（剩余{remaining}次验证机会）"
                            )
                        )
                        return


                    # 处理游戏问答消息
                    command_part = user_input.strip()  # 直接使用 plain_text
                    logger.info(f"处理游戏问答消息: '{command_part}'")

                    # 判断回答（是否问答）：按 judge_backend 走 LLM 或 Jev
                    logger.info(f"使用 {self.judge_backend} 判断游戏问答: '{command_part}'")
                    outcome = await self.judge_question(
                        command_part,
                        current_answer,
                        puzzle=(game or {}).get("puzzle", ""),
                        qa_history=(game or {}).get("qa_history", []),
                    )
                    # 问答历史存干净的判定；相关性百分比只出现在给玩家的消息里
                    reply = outcome.verdict

                    # 记录提问和回答
                    if game is not None:
                        history = game.setdefault("qa_history", [])
                        history.append({"question": command_part, "answer": reply})

                    # 更新问题计数
                    if question_limit is not None and game is not None:
                        game["question_count"] = game.get("question_count", 0) + 1
                        # 将判断结果、使用次数（和可选的相关性百分比）合并到一条消息中
                        combined_reply = outcome.display(
                            f"{game['question_count']}/{question_limit}",
                            self.show_relevance,
                        )
                        await event.send(event.plain_result(combined_reply))

                        if game["question_count"] >= question_limit:
                            await event.send(
                                event.plain_result(
                                    "❗️提问次数已用完，将进入验证环节。你有2次验证机会，请使用 /验证 <推理内容>。"
                                )
                            )
                    else:
                        # 如果没有问题限制，只发送判断结果
                        await event.send(
                            event.plain_result(outcome.display("", self.show_relevance))
                        )

                    # 重置超时时间
                    controller.keep(timeout=self.game_timeout, reset_timeout=True)

                except Exception as e:
                    logger.error(f"会话控制内部错误: {e}")
                    await event.send(event.plain_result(f"游戏处理过程中发生错误：{e}"))
                    # 如果发生错误，结束游戏
                    self.game_state.end_game(group_id)
                    controller.stop()

            try:
                # 使用群 ID 限制会话范围，避免多个群并发时互相触发
                await game_session_waiter(event, session_filter=GroupSessionFilter(group_id))
            except TimeoutError:
                game = self.game_state.get_game(group_id)
                if game:
                    await event.send(
                        event.plain_result(
                            f"⏰ 游戏超时！\n\n📖 完整故事：{game['answer']}\n\n游戏结束！"
                        )
                    )
                    self.game_state.end_game(group_id)
            except Exception as e:
                logger.error(f"游戏会话错误: {e}")
                await event.send(event.plain_result(f"游戏过程中发生错误：{e}"))
                self.game_state.end_game(group_id)
        except Exception as e:
            logger.error(f"启动游戏会话失败: {e}")
            await event.send(event.plain_result(f"启动游戏会话失败：{e}"))

    def _is_at_bot(self, event: AstrMessageEvent) -> bool:
        """检查消息是否@了bot"""

        bot_id = str(event.get_self_id())
        for comp in event.message_obj.message:
            if isinstance(comp, At) and str(comp.qq) == bot_id:
                return True
        return False

    async def get_story_by_strategy(self, strategy: str) -> Optional[Tuple[str, str]]:
        """根据策略获取故事，返回 (puzzle, answer) 或 None"""
        import random

        self._ensure_story_storages()

        if strategy == "network_first":
            # 策略1：优先网络题库 -> 本地存储库 -> 自定义题库 -> LLM现场生成

            # 1. 检查网络题库
            story = self.online_story_storage.get_story()
            if story:
                return story

            # 2. 检查本地存储库
            story = self.local_story_storage.get_story()
            if story:
                return story

            # 3. 检查自定义题库
            story = self.custom_story_storage.get_story()
            if story:
                return story

            # 4. LLM现场生成
            return await self.generate_story_with_llm()

        elif strategy == "local_first":
            # 策略2：优先本地存储库 -> 网络题库 -> 自定义题库 -> LLM现场生成

            # 1. 检查本地存储库
            story = self.local_story_storage.get_story()
            if story:
                return story

            # 2. 检查网络题库
            story = self.online_story_storage.get_story()
            if story:
                return story

            # 3. 检查自定义题库
            story = self.custom_story_storage.get_story()
            if story:
                return story

            # 4. LLM现场生成
            return await self.generate_story_with_llm()

        elif strategy == "custom_first":
            # 策略3：优先自定义题库 -> 本地存储库 -> 网络题库 -> LLM现场生成

            # 1. 检查自定义题库
            story = self.custom_story_storage.get_story()
            if story:
                return story

            # 2. 检查本地存储库
            story = self.local_story_storage.get_story()
            if story:
                return story

            # 3. 检查网络题库
            story = self.online_story_storage.get_story()
            if story:
                return story

            # 4. LLM现场生成
            return await self.generate_story_with_llm()

        elif strategy == "random":
            # 策略3：随机选择网络题库、本地存储库或自定义题库，失败时使用LLM现场生成

            # 随机决定这次从哪个题库获取
            choice = random.choice(["network", "local", "custom"])
            if choice == "network":
                # 参考策略1的网络题库逻辑
                story = self.online_story_storage.get_story()
                if story:
                    return story

                story = self.local_story_storage.get_story()
                if story:
                    return story

                story = self.custom_story_storage.get_story()
                if story:
                    return story

                return await self.generate_story_with_llm()
            elif choice == "local":
                # 参考策略2的本地存储库逻辑
                story = self.local_story_storage.get_story()
                if story:
                    return story

                story = self.online_story_storage.get_story()
                if story:
                    return story

                story = self.custom_story_storage.get_story()
                if story:
                    return story

                return await self.generate_story_with_llm()
            else:  # custom
                # 优先自定义题库
                story = self.custom_story_storage.get_story()
                if story:
                    return story

                story = self.local_story_storage.get_story()
                if story:
                    return story

                story = self.online_story_storage.get_story()
                if story:
                    return story

                return await self.generate_story_with_llm()

        return None

    async def get_story_by_index(self, source_type: str, index: int) -> Optional[Tuple[str, str]]:
        """根据索引获取特定故事
        
        Args:
            source_type: "network" - 网络题库, "current" - 当前策略题库, "custom" - 自定义题库
            index: 题目索引（从0开始）
        
        Returns:
            (puzzle, answer) 或 None
        """
        self._ensure_story_storages()
        
        if source_type == "network":
            # 从网络题库获取指定索引的故事
            if index < 0 or index >= len(self.online_story_storage.stories):
                return None
            
            story = self.online_story_storage.stories[index]
            # 标记为已使用
            with self.online_story_storage.lock:
                self.online_story_storage.used_indexes.add(index)
                self.online_story_storage.save_usage_record()
            
            logger.info(f"从网络题库获取指定故事，索引: {index}")
            return story["puzzle"], story["answer"]
            
        elif source_type == "custom":
            # 从自定义题库获取指定索引的故事
            if index < 0 or index >= len(self.custom_story_storage.stories):
                return None
            
            story = self.custom_story_storage.stories[index]
            # 标记为已使用
            with self.custom_story_storage.lock:
                self.custom_story_storage.used_indexes.add(index)
                self.custom_story_storage.save_usage_record()
            
            logger.info(f"从自定义题库获取指定故事，索引: {index}")
            return story["puzzle"], story["answer"]
            
        elif source_type == "current":
            # 根据当前策略获取指定索引的故事
            strategy = self.puzzle_source_strategy
            
            if strategy == "network_first":
                # 优先检查网络题库
                if index < len(self.online_story_storage.stories):
                    story = self.online_story_storage.stories[index]
                    with self.online_story_storage.lock:
                        self.online_story_storage.used_indexes.add(index)
                        self.online_story_storage.save_usage_record()
                    logger.info(f"从网络题库获取指定故事，索引: {index}")
                    return story["puzzle"], story["answer"]
                
                # 然后检查本地存储库
                local_index = index - len(self.online_story_storage.stories)
                if local_index >= 0 and local_index < len(self.local_story_storage.stories):
                    story = self.local_story_storage.stories[local_index]
                    with self.local_story_storage.lock:
                        self.local_story_storage.used_indexes.add(local_index)
                        self.local_story_storage.save_usage_record()
                    logger.info(f"从本地存储库获取指定故事，索引: {local_index}")
                    return story["puzzle"], story["answer"]
                
                # 然后检查自定义题库
                custom_index = local_index - len(self.local_story_storage.stories)
                if custom_index >= 0 and custom_index < len(self.custom_story_storage.stories):
                    story = self.custom_story_storage.stories[custom_index]
                    with self.custom_story_storage.lock:
                        self.custom_story_storage.used_indexes.add(custom_index)
                        self.custom_story_storage.save_usage_record()
                    logger.info(f"从自定义题库获取指定故事，索引: {custom_index}")
                    return story["puzzle"], story["answer"]
                
                # 超出范围，返回None
                return None
                
            elif strategy == "local_first":
                # 优先检查本地存储库
                if index < len(self.local_story_storage.stories):
                    story = self.local_story_storage.stories[index]
                    with self.local_story_storage.lock:
                        self.local_story_storage.used_indexes.add(index)
                        self.local_story_storage.save_usage_record()
                    logger.info(f"从本地存储库获取指定故事，索引: {index}")
                    return story["puzzle"], story["answer"]
                
                # 然后检查网络题库
                network_index = index - len(self.local_story_storage.stories)
                if network_index >= 0 and network_index < len(self.online_story_storage.stories):
                    story = self.online_story_storage.stories[network_index]
                    with self.online_story_storage.lock:
                        self.online_story_storage.used_indexes.add(network_index)
                        self.online_story_storage.save_usage_record()
                    logger.info(f"从网络题库获取指定故事，索引: {network_index}")
                    return story["puzzle"], story["answer"]
                
                # 然后检查自定义题库
                custom_index = network_index - len(self.online_story_storage.stories)
                if custom_index >= 0 and custom_index < len(self.custom_story_storage.stories):
                    story = self.custom_story_storage.stories[custom_index]
                    with self.custom_story_storage.lock:
                        self.custom_story_storage.used_indexes.add(custom_index)
                        self.custom_story_storage.save_usage_record()
                    logger.info(f"从自定义题库获取指定故事，索引: {custom_index}")
                    return story["puzzle"], story["answer"]
                
                # 超出范围，返回None
                return None
                
            elif strategy == "custom_first":
                # 优先检查自定义题库
                if index < len(self.custom_story_storage.stories):
                    story = self.custom_story_storage.stories[index]
                    with self.custom_story_storage.lock:
                        self.custom_story_storage.used_indexes.add(index)
                        self.custom_story_storage.save_usage_record()
                    logger.info(f"从自定义题库获取指定故事，索引: {index}")
                    return story["puzzle"], story["answer"]
                
                # 然后检查本地存储库
                local_index = index - len(self.custom_story_storage.stories)
                if local_index >= 0 and local_index < len(self.local_story_storage.stories):
                    story = self.local_story_storage.stories[local_index]
                    with self.local_story_storage.lock:
                        self.local_story_storage.used_indexes.add(local_index)
                        self.local_story_storage.save_usage_record()
                    logger.info(f"从本地存储库获取指定故事，索引: {local_index}")
                    return story["puzzle"], story["answer"]
                
                # 然后检查网络题库
                network_index = local_index - len(self.local_story_storage.stories)
                if network_index >= 0 and network_index < len(self.online_story_storage.stories):
                    story = self.online_story_storage.stories[network_index]
                    with self.online_story_storage.lock:
                        self.online_story_storage.used_indexes.add(network_index)
                        self.online_story_storage.save_usage_record()
                    logger.info(f"从网络题库获取指定故事，索引: {network_index}")
                    return story["puzzle"], story["answer"]
                
                # 超出范围，返回None
                return None
                
            elif strategy == "random":
                # 对于随机策略，我们无法准确知道索引对应哪个题库
                # 这里我们按顺序检查：先网络题库，再本地存储库，最后自定义题库
                if index < len(self.online_story_storage.stories):
                    story = self.online_story_storage.stories[index]
                    with self.online_story_storage.lock:
                        self.online_story_storage.used_indexes.add(index)
                        self.online_story_storage.save_usage_record()
                    logger.info(f"从网络题库获取指定故事，索引: {index}")
                    return story["puzzle"], story["answer"]
                
                local_index = index - len(self.online_story_storage.stories)
                if local_index >= 0 and local_index < len(self.local_story_storage.stories):
                    story = self.local_story_storage.stories[local_index]
                    with self.local_story_storage.lock:
                        self.local_story_storage.used_indexes.add(local_index)
                        self.local_story_storage.save_usage_record()
                    logger.info(f"从本地存储库获取指定故事，索引: {local_index}")
                    return story["puzzle"], story["answer"]
                
                custom_index = local_index - len(self.local_story_storage.stories)
                if custom_index >= 0 and custom_index < len(self.custom_story_storage.stories):
                    story = self.custom_story_storage.stories[custom_index]
                    with self.custom_story_storage.lock:
                        self.custom_story_storage.used_indexes.add(custom_index)
                        self.custom_story_storage.save_usage_record()
                    logger.info(f"从自定义题库获取指定故事，索引: {custom_index}")
                    return story["puzzle"], story["answer"]
                
                # 超出范围，返回None
                return None
        
        return None

    async def _handle_game_status_in_session(
            self, event: AstrMessageEvent, group_id: str
    ):
        """在会话控制中处理游戏状态查询逻辑"""
        try:

            if self.game_state.is_game_active(group_id):
                game = self.game_state.get_game(group_id)
                difficulty = game.get("difficulty", "普通")
                question_count = game.get("question_count", 0)
                question_limit = game.get("question_limit")
                hint_count = game.get("hint_count", 0)
                hint_limit = game.get("hint_limit")

                question_info = f"{question_count}/{question_limit}" if question_limit else f"{question_count}/∞"
                hint_info = f"{hint_count}/{hint_limit}" if hint_limit else "不可用"

                await event.send(
                    event.plain_result(
                        f"🎮 当前有活跃的海龟汤游戏\n📖 题面：{game['puzzle']}\n🎯 难度：{difficulty}\n❓ 提问：{question_info}\n💡 提示：{hint_info}"
                    )
                )
            else:
                await event.send(
                    event.plain_result(
                        "🎮 当前没有活跃的海龟汤游戏\n💡 使用 /汤 开始新游戏"
                    )
                )

        except Exception as e:
            logger.error(f"会话游戏状态查询失败: {e}")
            await event.send(event.plain_result(f"查询游戏状态时发生错误：{e}"))

    async def _handle_force_end_in_session(
            self, event: AstrMessageEvent, group_id: str
    ):
        """在会话控制中处理强制结束游戏逻辑"""
        try:

            if self.game_state.end_game(group_id):
                await event.send(event.plain_result("✅ 已强制结束当前海龟汤游戏"))
            else:
                await event.send(event.plain_result("❌ 当前没有活跃的游戏需要结束"))

        except Exception as e:
            logger.error(f"会话强制结束失败: {e}")
            await event.send(event.plain_result(f"强制结束游戏时发生错误：{e}"))

    async def _handle_view_history_in_session(
            self, event: AstrMessageEvent, group_id: str
    ):
        """在会话控制中处理查看历史记录逻辑"""
        try:


            game = self.game_state.get_game(group_id)
            if not game:
                await event.send(event.plain_result("无法获取游戏状态"))
                return

            history = game.get("qa_history", [])

            if not history:
                await event.send(event.plain_result("目前还没有人提问哦~"))
                return

            lines = ["📋 提问记录："]
            for idx, item in enumerate(history, 1):
                lines.append(f"{idx}. 问：{item['question']}\n   答：{item['answer']}")

            response = "\n".join(lines)
            await event.send(event.plain_result(response))

        except Exception as e:
            logger.error(f"会话查看历史失败: {e}")
            await event.send(event.plain_result(f"查看历史记录时发生错误：{e}"))

    async def _build_hint_result(
            self, event: AstrMessageEvent, group_id: str
    ) -> Optional[MessageEventResult]:
        """生成提示结果，供指令或会话控制调用"""
        if not group_id:
            return event.plain_result("提示功能只能在群聊中使用")

        game = self.game_state.get_game(group_id)
        if not game:
            return event.plain_result("当前没有活跃的海龟汤游戏")

        hint_limit = game.get("hint_limit")
        hint_count = game.get("hint_count", 0)
        if hint_limit == 0:
            return event.plain_result("当前难度不可使用提示")
        if hint_limit is not None and hint_count >= hint_limit:
            return event.plain_result("提示次数已用完")

        qa_history = game.get("qa_history", [])
        if not qa_history:
            return event.plain_result("请先进行提问后再请求提示")

        hint_history = game.get("hint_history", [])
        allow_list = self.build_allow_list(game["puzzle"], qa_history)

        hint = await self.generate_hint(
            game["puzzle"], game["answer"], qa_history, hint_history, allow_list
        )
        game["hint_count"] = hint_count + 1
        game["hint_history"] = hint_history + [hint]
        suffix = ""
        if hint_limit is not None:
            suffix = f"（{game['hint_count']}/{hint_limit}）"
        return event.plain_result(f"提示：{hint}{suffix}")

    async def _handle_verification_in_session(
            self, event: AstrMessageEvent, user_guess: str, answer: str
    ):
        """在会话控制中处理验证逻辑"""
        try:

            # 验证用户推理
            result = await self.verify_user_guess(user_guess, answer)

            group_id = event.get_group_id()
            game = self.game_state.get_game(group_id) if group_id else None
            accept_levels = (
                game.get("accept_levels", ["完全还原", "核心推理正确"])
                if game
                else ["完全还原", "核心推理正确"]
            )
            is_correct = result.level in accept_levels

            # 返回验证结果
            response = f"等级：{result.level}\n评价：{result.comment}"
            await event.send(event.plain_result(response))

            if is_correct:
                await event.send(
                    event.plain_result(
                        f"🎉 恭喜！你猜中了！\n\n📖 完整故事：{answer}\n\n游戏结束！"
                    )
                )
                if group_id:
                    self.game_state.end_game(group_id)
                return

            if (
                    game
                    and game.get("question_limit") is not None
                    and game.get("question_count", 0) >= game.get("question_limit")
            ):
                game["verification_attempts"] = game.get("verification_attempts", 0) + 1
                remaining = 2 - game["verification_attempts"]
                if remaining > 0:
                    await event.send(
                        event.plain_result(
                            f"❌ 验证未通过，你还有 {remaining} 次机会。"
                        )
                    )
                else:
                    await event.send(
                        event.plain_result(
                            f"❌ 验证未通过。\n\n📖 完整故事：{answer}\n\n游戏结束！"
                        )
                    )
                    self.game_state.end_game(group_id)

        except Exception as e:
            logger.error(f"会话验证失败: {e}")
            await event.send(event.plain_result(f"验证过程中发生错误：{e}"))

    # 📊 游戏状态查询
    @filter.command("汤状态")
    async def check_game_status(self, event: AstrMessageEvent):
        """查看当前游戏状态"""
        group_id = event.get_group_id()

        if not group_id:
            yield event.plain_result("此功能只能在群聊中使用")
            return

        if self.game_state.is_game_active(group_id):
            game = self.game_state.get_game(group_id)
            difficulty = game.get("difficulty", "普通")
            question_count = game.get("question_count", 0)
            question_limit = game.get("question_limit")
            hint_count = game.get("hint_count", 0)
            hint_limit = game.get("hint_limit")

            question_info = f"{question_count}/{question_limit}" if question_limit else f"{question_count}/∞"
            hint_info = f"{hint_count}/{hint_limit}" if hint_limit else "不可用"

            yield event.plain_result(
                f"🎮 当前有活跃的海龟汤游戏\n📖 题面：{game['puzzle']}\n🎯 难度：{difficulty}\n❓ 提问：{question_info}\n💡 提示：{hint_info}"
            )
        else:
            yield event.plain_result(
                "🎮 当前没有活跃的海龟汤游戏\n💡 使用 /汤 开始新游戏"
            )

    @filter.command("查看")
    async def view_question_history(self, event: AstrMessageEvent):
        """查看当前已提问的问题及回答"""
        group_id = event.get_group_id()

        if not group_id:
            yield event.plain_result("此功能只能在群聊中使用")
            return
        if not self.game_state.is_game_active(group_id):
            yield event.plain_result("当前没有活跃的海龟汤游戏")
            return
        game = self.game_state.get_game(group_id)
        history = game.get("qa_history", []) if game else []
        if not history:
            yield event.plain_result("目前还没有人提问哦~")
            return
        lines = ["📋 提问记录："]
        for idx, item in enumerate(history, 1):
            lines.append(f"{idx}. 问：{item['question']}\n   答：{item['answer']}")
        yield event.plain_result("\n".join(lines))

    # 🆘 强制结束游戏（管理员功能）
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("强制结束")
    async def force_end_game(self, event: AstrMessageEvent):
        """强制结束当前游戏（仅管理员）"""
        group_id = event.get_group_id()

        if not group_id:
            yield event.plain_result("此功能只能在群聊中使用")
            return

        if self.game_state.end_game(group_id):
            yield event.plain_result("✅ 已强制结束当前海龟汤游戏")
        else:
            yield event.plain_result("❌ 当前没有活跃的游戏需要结束")

    # 📚 备用故事管理指令
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("备用开始")
    async def start_backup_generation(self, event: AstrMessageEvent):
        """开始生成备用故事（仅管理员）"""

        if self.auto_generating:
            yield event.plain_result("⚠️ 备用故事生成已在运行中")
            return

        # 检查存储库是否已满
        self._ensure_story_storages()
        storage_info = self.local_story_storage.get_storage_info()
        if storage_info["available"] <= 0:
            yield event.plain_result("⚠️ 存储库已满，无法生成更多故事")
            return

        self.auto_generating = True
        asyncio.create_task(self._auto_generate_loop())
        yield event.plain_result(
            f"✅ 开始生成备用故事，存储库状态: {storage_info['total']}/{storage_info['max_size']}"
        )

    # 🔒 全局指令拦截器 - 当正在生成时提醒用户
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def global_command_interceptor(self, event: AstrMessageEvent):
        """全局指令拦截器，当正在生成备用故事时提醒用户"""
        # 检查是否有活跃游戏，如果有活跃游戏，不在这里处理
        group_id = event.get_group_id()
        if group_id and self.game_state.is_game_active(group_id):
            # 有活跃游戏，让会话控制处理
            return

        # 如果正在生成备用故事，且不是 /备用结束 指令，则提醒用户
        if self.auto_generating:
            user_input = event.message_str.strip()
            # 只拦截非本插件的指令，避免阻断自己的指令
            if (
                    user_input.startswith("/")
                    and not user_input.startswith("/备用结束")
                    and not user_input.startswith("/汤")
                    and not user_input.startswith("/揭晓")
                    and not user_input.startswith("/验证")
                    and not user_input.startswith("/汤状态")
                    and not user_input.startswith("/强制结束")
                    and not user_input.startswith("/备用开始")
                    and not user_input.startswith("/备用状态")
                    and not user_input.startswith("/汤配置")
                    and not user_input.startswith("/重置题库")
                    and not user_input.startswith("/题库详情")
                    and not user_input.startswith("/查看")
                    and not user_input.startswith("/提示")
            ):
                yield event.plain_result(
                    "⚠️ 系统正在生成备用故事，请稍后再试或使用 /备用结束 停止生成"
                )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("备用结束")
    async def stop_backup_generation(self, event: AstrMessageEvent):
        """停止生成备用故事（仅管理员）"""

        if not self.auto_generating:
            yield event.plain_result("⚠️ 备用故事生成未在运行")
            return

        self.auto_generating = False
        yield event.plain_result("✅ 已停止生成备用故事，正在完成当前生成...")

    @filter.command("备用状态")
    async def check_backup_status(self, event: AstrMessageEvent):
        """查看备用故事状态"""
        self._ensure_story_storages()
        storage_info = self.local_story_storage.get_storage_info()
        online_info = self.online_story_storage.get_storage_info()
        status = "🟢 运行中" if self.auto_generating else "🔴 已停止"


        # 检查存储库是否已满
        storage_full_warning = ""
        if storage_info["available"] <= 0:
            storage_full_warning = "\n⚠️ 本地存储库已满，自动生成已停止"

        message = (
            f"📚 备用故事状态：\n"
            f"• 生成状态：{status}\n"
            f"• 本地存储库：{storage_info['total']}/{storage_info['max_size']}\n"
            f"• 已使用题目：{storage_info['used']}\n"
            f"• 剩余题目：{storage_info['remaining']}\n"
            f"• 可用空间：{storage_info['available']}\n"
            f"• 网络题库：{online_info['total']} 个 (已用: {online_info['used']}, 剩余: {online_info['available']})\n"
            f"• 自动生成时间：{self.auto_generate_start}:00-{self.auto_generate_end}:00{storage_full_warning}"
        )

        yield event.plain_result(message)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("重置题库")
    async def reset_story_storage(self, event: AstrMessageEvent):
        """重置题库使用记录（仅管理员）"""

        self._ensure_story_storages()

        # 重置网络题库使用记录
        self.online_story_storage.reset_usage()
        online_info = self.online_story_storage.get_storage_info()

        # 重置本地存储库使用记录
        self.local_story_storage.reset_usage()
        local_info = self.local_story_storage.get_storage_info()


        message = (
            f"✅ 题库使用记录已重置！\n"
            f"• 网络题库：{online_info['total']} 个谜题 (已重置)\n"
            f"• 本地存储库：{local_info['total']} 个谜题 (已重置)\n"
            f"• 所有题目现在都可以重新使用"
        )

        yield event.plain_result(message)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("题库详情")
    async def show_storage_details(self, event: AstrMessageEvent):
        """查看题库详细使用记录（仅管理员）"""

        # 确保题库已初始化
        self._ensure_story_storages()

        # 获取网络题库详细信息
        online_info = self.online_story_storage.get_storage_info()
        online_usage = self.online_story_storage.get_usage_info()

        # 获取本地存储库详细信息
        local_info = self.local_story_storage.get_storage_info()
        local_usage = self.local_story_storage.get_usage_info()


        # 安全计算使用率，避免除零错误
        online_usage_rate = (
            (online_info["used"] / online_info["total"] * 100)
            if online_info["total"] > 0
            else 0.0
        )
        local_usage_rate = (
            (local_info["used"] / local_info["total"] * 100)
            if local_info["total"] > 0
            else 0.0
        )

        message = (
            f"📊 题库详细使用记录：\n\n"
            f"🌐 网络题库：\n"
            f"• 总数：{online_info['total']} 个谜题\n"
            f"• 已使用：{online_info['used']} 个\n"
            f"• 剩余：{online_info['available']} 个\n"
            f"• 使用率：{online_usage_rate:.1f}%\n"
            f"• 已用索引：{online_usage['used_indexes'][:10]}{'...' if len(online_usage['used_indexes']) > 10 else ''}\n\n"
            f"💾 本地存储库：\n"
            f"• 总数：{local_info['total']} 个谜题\n"
            f"• 已使用：{local_info['used']} 个\n"
            f"• 剩余：{local_info['remaining']} 个\n"
            f"• 使用率：{local_usage_rate:.1f}%\n"
            f"• 已用索引：{local_usage['used_indexes'][:10]}{'...' if len(local_usage['used_indexes']) > 10 else ''}"
        )

        yield event.plain_result(message)

    @filter.command("提示")
    async def hint_command(self, event: AstrMessageEvent):
        """根据当前所有提问记录提供方向性提示"""
        result = await self._build_hint_result(event, event.get_group_id())
        if result:
            yield result

    # 🔍 验证指令（仅在非游戏会话时处理）
    @filter.command("验证")
    async def verify_user_guess_command(self, event: AstrMessageEvent, user_guess: str):
        """验证用户推理（仅在非游戏会话时处理）"""
        group_id = event.get_group_id()

        if not group_id:
            yield event.plain_result("验证功能只能在群聊中使用")
            return

        # 检查是否有活跃游戏，如果有活跃游戏，说明在会话控制中，不在这里处理
        if self.game_state.is_game_active(group_id):
            # 阻止事件继续传播，避免被会话控制系统重复处理
            await event.block()
            return
        # 只有在没有活跃游戏时才在这里处理（用于游戏外的验证）
        yield event.plain_result("当前没有活跃的海龟汤游戏，请使用 /汤 开始新游戏")

    # ⚙️ 查看当前配置
    @filter.command("汤配置")
    async def show_config(self, event: AstrMessageEvent):
        """查看当前插件配置"""

        # 确保题库已初始化
        self._ensure_story_storages()

        local_info = self.local_story_storage.get_storage_info()
        online_info = self.online_story_storage.get_storage_info()

        # 获取策略的中文描述
        strategy_names = {
            "network_first": "优先网络题库→本地存储库→LLM生成",
            "local_first": "优先本地存储库→网络题库→LLM生成",
            "custom_first": "优先自定义题库→本地存储库→LLM生成",
            "random": "随机从网络、本地或自定义题库中选择",
        }
        strategy_name = strategy_names.get(
            self.puzzle_source_strategy, self.puzzle_source_strategy
        )

        # 检查存储库是否已满
        storage_full_warning = ""
        if local_info["available"] <= 0:
            storage_full_warning = "\n⚠️ 本地存储库已满，自动生成已停止"

        judge_backend_names = {
            JUDGE_BACKEND_LLM: "LLM",
            JUDGE_BACKEND_JEV: "Jev 决策模型",
            JUDGE_BACKEND_HYBRID: "Jev 优先（置信度不足回落 LLM）",
        }
        jev_enabled = bool((self.jev_judge_config or {}).get("enabled"))
        jev_line = ""
        if self.judge_backend != JUDGE_BACKEND_LLM:
            language = jev_judge.resolve_language(
                self.jev_judge_config.get("prompt_language"), "", "", ""
            )
            language_names = {"zh": "中文", "en": "英文"}
            proxy_url = str(self.jev_judge_config.get("proxy_url") or "").strip()
            target = (
                f"{proxy_url} · 协议 "
                f"{self.jev_judge_config.get('proxy_protocol') or 'systemone'}"
                if proxy_url
                else "未配置网关"
            )
            jev_line = (
                f"• Jev 决策模型：{'已启用' if jev_enabled else '未启用（将回退 LLM）'}"
                f"（{target}）\n"
                f"• Jev 提示词：{jev_judge.PROMPT_VERSION}"
                f"（语言 {language_names.get(language, language)}）\n"
            )
        config_info = (
            f"⚙️ 海龟汤插件配置：\n"
            f"• 生成谜题 LLM：{self.generate_llm_provider_id or '默认'}\n"
            f"• 判断问答 LLM：{self.judge_llm_provider_id or '默认'}\n"
            f"• 判断后端：{judge_backend_names.get(self.judge_backend, self.judge_backend)}\n"
            f"{jev_line}"
            f"• 游戏超时：{self.game_timeout} 秒\n"
            f"• 网络题库：{online_info['total']} 个谜题 (已用: {online_info['used']}, 剩余: {online_info['available']})\n"
            f"• 本地存储库：{local_info['total']}/{local_info['max_size']} (已用: {local_info['used']}, 剩余: {local_info['remaining']})\n"
            f"• 自动生成时间：{self.auto_generate_start}:00-{self.auto_generate_end}:00\n"
            f"• 谜题来源策略：{strategy_name}{storage_full_warning}"
        )
        yield event.plain_result(config_info)

    # ➕ 添加自定义海龟汤
    @filter.command("添加海龟汤")
    async def add_custom_soupai(self, event: AstrMessageEvent, content: str):
        """添加自定义海龟汤故事，格式: /添加海龟汤 <汤面>|<汤底>"""
        
        # 确保自定义存储库已初始化
        self._ensure_story_storages()
        
        # 解析内容格式: 汤面|汤底
        if "|" not in content:
            yield event.plain_result("❌ 格式错误！请使用格式: /添加海龟汤 <汤面>|<汤底>")
            return
        
        puzzle, answer = content.split("|", 1)
        puzzle = puzzle.strip()
        answer = answer.strip()
        
        if not puzzle or not answer:
            yield event.plain_result("❌ 汤面和汤底都不能为空！")
            return
        
        # 添加故事到自定义存储库
        success = self.custom_story_storage.add_story(puzzle, answer)
        
        if success:
            # 获取添加后的故事索引
            story_index = len(self.custom_story_storage.stories) - 1
            yield event.plain_result(
                f"✅ 添加成功！海龟汤编号: {story_index}\n\n"
                f"📖 汤面: {puzzle}\n"
                f"📖 汤底: {answer}"
            )
        else:
            yield event.plain_result("❌ 添加失败，请重试")
