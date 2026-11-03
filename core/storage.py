"""题库存储：网络题库 / 本地存储库 / 自定义故事库。

三者共用 ``ThreadSafeStoryStorage`` 的使用记录（哪些题已经出过），
并保证并发取题时不会重复出同一道题。
"""

import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from astrbot.api import logger


class ThreadSafeStoryStorage:
    """线程安全的题库管理基类，支持持久化使用记录"""

    def __init__(self, storage_name: str, data_path=None):
        self.storage_name = storage_name
        self.data_path = data_path
        self.used_indexes: set[int] = set()
        self.lock = threading.Lock()  # 线程锁
        self.usage_file = (
            self.data_path / f"{storage_name}_usage.json" if self.data_path else None
        )
        self.load_usage_record()

    def load_usage_record(self):
        """从文件加载使用记录"""
        if not self.usage_file:
            self.used_indexes = set()
            return

        try:
            if self.usage_file.exists():
                with open(self.usage_file, "r", encoding="utf-8") as f:
                    self.used_indexes = set(json.load(f))
                logger.info(
                    f"从 {self.usage_file} 加载了 {len(self.used_indexes)} 个使用记录"
                )
            else:
                self.used_indexes = set()
                logger.info(
                    f"使用记录文件不存在，创建新的记录: {self.usage_file}"
                )
        except Exception as e:
            logger.error(f"加载使用记录失败: {e}")
            self.used_indexes = set()

    def save_usage_record(self):
        """保存使用记录到文件"""
        if not self.usage_file:
            return

        try:
            self.usage_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.usage_file, "w", encoding="utf-8") as f:
                json.dump(list(self.used_indexes), f, ensure_ascii=False, indent=2)
            logger.info(
                f"保存了 {len(self.used_indexes)} 个使用记录到 {self.usage_file}"
            )
        except Exception as e:
            logger.error(f"保存使用记录失败: {e}")

    def reset_usage(self):
        """重置使用记录"""
        with self.lock:
            self.used_indexes.clear()
            self.save_usage_record()
            logger.info(f"{self.storage_name} 使用记录已重置")

    def get_usage_info(self) -> Dict:
        """获取使用记录信息"""
        with self.lock:
            return {
                "used": len(self.used_indexes),
                "used_indexes": list(self.used_indexes),
            }


# Jev 决策后端


class NetworkSoupaiStorage(ThreadSafeStoryStorage):
    def __init__(self, network_file: str, data_path=None):
        # 初始化基类
        super().__init__("network_soupai", data_path)
        self.network_file = network_file
        self.stories: List[Dict] = []
        self.load_stories()

    def load_stories(self):
        """从文件加载网络海龟汤故事"""
        try:
            if os.path.exists(self.network_file):
                with open(self.network_file, "r", encoding="utf-8") as f:
                    self.stories = json.load(f)
                logger.info(
                    f"从 {self.network_file} 加载了 {len(self.stories)} 个网络海龟汤故事"
                )
            else:
                self.stories = []
                logger.warning(f"网络海龟汤文件不存在: {self.network_file}")
        except Exception as e:
            logger.error(f"加载网络海龟汤失败: {e}")
            self.stories = []

    def get_story(self) -> Optional[Tuple[str, str]]:
        """从网络题库获取一个故事，避免重复（线程安全）"""
        if not self.stories:
            return None

        with self.lock:
            # 获取所有可用的索引（排除已使用的）
            available_indexes = [
                i for i in range(len(self.stories)) if i not in self.used_indexes
            ]

            # 如果没有可用题目，清空已用记录，重新开始一轮
            if not available_indexes:
                logger.info("网络题库已全部使用完毕，清空记录重新开始")
                self.used_indexes.clear()
                available_indexes = list(range(len(self.stories)))
                # 立即保存重置后的状态
                self.save_usage_record()

            # 从可用索引中随机选择一个
            import random

            selected = random.choice(available_indexes)
            self.used_indexes.add(selected)

            # 保存使用记录
            self.save_usage_record()

            story = self.stories[selected]
            logger.info(
                f"从网络题库获取故事，索引: {selected}, 已使用: {len(self.used_indexes)}/{len(self.stories)}"
            )
            return story["puzzle"], story["answer"]

    def get_storage_info(self) -> Dict:
        """获取网络题库信息"""
        usage_info = self.get_usage_info()
        return {
            "total": len(self.stories),
            "available": len(self.stories) - usage_info["used"],
            "used": usage_info["used"],
        }


# 存储库管理


class LocalSoupaiStorage(ThreadSafeStoryStorage):
    def __init__(self, storage_file: str, max_size: int = 50, data_path=None):
        # 初始化基类
        super().__init__("storage_soupai", data_path)
        self.storage_file = storage_file
        self.max_size = max_size
        self.stories: List[Dict] = []
        self.load_stories()

    def load_stories(self):
        """从文件加载故事"""
        try:
            storage_path = (
                self.storage_file
                if isinstance(self.storage_file, str)
                else str(self.storage_file)
            )
            if os.path.exists(storage_path):
                with open(storage_path, "r", encoding="utf-8") as f:
                    self.stories = json.load(f)
                logger.info(f"从 {storage_path} 加载了 {len(self.stories)} 个故事")
            else:
                self.stories = []
                logger.info("存储库文件不存在，创建新的存储库")
        except Exception as e:
            logger.error(f"加载故事失败: {e}")
            self.stories = []

    def save_stories(self):
        """保存故事到文件"""
        try:
            storage_path = (
                self.storage_file
                if isinstance(self.storage_file, str)
                else str(self.storage_file)
            )
            # 确保目录存在
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
            with open(storage_path, "w", encoding="utf-8") as f:
                json.dump(self.stories, f, ensure_ascii=False, indent=2)
            logger.info(f"保存了 {len(self.stories)} 个故事到 {storage_path}")
        except Exception as e:
            logger.error(f"保存故事失败: {e}")

    def add_story(self, puzzle: str, answer: str) -> bool:
        """添加故事到存储库"""
        with self.lock:
            if len(self.stories) >= self.max_size:
                # 移除最旧的故事
                self.stories.pop(0)
                logger.info("存储库已满，移除最旧的故事")

            story = {
                "puzzle": puzzle,
                "answer": answer,
                "created_at": datetime.now().isoformat(),
            }
            self.stories.append(story)
            self.save_stories()
            logger.info(f"添加新故事到存储库，当前存储库大小: {len(self.stories)}")
            return True

    def get_story(self) -> Optional[Tuple[str, str]]:
        """从存储库获取一个故事，避免重复（线程安全）"""
        if not self.stories:
            return None

        with self.lock:
            # 获取所有可用的索引（排除已使用的）
            available_indexes = [
                i for i in range(len(self.stories)) if i not in self.used_indexes
            ]

            # 如果没有可用题目，清空已用记录，重新开始一轮
            if not available_indexes:
                logger.info("本地存储库已全部使用完毕，清空记录重新开始")
                self.used_indexes.clear()
                available_indexes = list(range(len(self.stories)))
                # 立即保存重置后的状态
                self.save_usage_record()

            # 从可用索引中随机选择一个
            import random

            selected = random.choice(available_indexes)
            self.used_indexes.add(selected)

            # 保存使用记录
            self.save_usage_record()

            story = self.stories[selected]
            logger.info(
                f"从本地存储库获取故事，索引: {selected}, 已使用: {len(self.used_indexes)}/{len(self.stories)}"
            )
            return story["puzzle"], story["answer"]

    def get_storage_info(self) -> Dict:
        """获取存储库信息"""
        usage_info = self.get_usage_info()
        return {
            "total": len(self.stories),
            "max_size": self.max_size,
            "available": self.max_size - len(self.stories),
            "used": usage_info["used"],
            "remaining": len(self.stories) - usage_info["used"],
        }


# 自定义海龟汤存储


class CustomSoupaiStorage(ThreadSafeStoryStorage):
    def __init__(self, storage_file: str, data_path=None):
        # 初始化基类
        super().__init__("custom_soupai", data_path)
        self.storage_file = storage_file
        self.stories: List[Dict] = []
        self.load_stories()

    def load_stories(self):
        """从文件加载自定义故事"""
        try:
            storage_path = (
                self.storage_file
                if isinstance(self.storage_file, str)
                else str(self.storage_file)
            )
            if os.path.exists(storage_path):
                with open(storage_path, "r", encoding="utf-8") as f:
                    self.stories = json.load(f)
                logger.info(f"从 {storage_path} 加载了 {len(self.stories)} 个自定义海龟汤故事")
            else:
                self.stories = []
                logger.info("自定义海龟汤文件不存在，创建新的存储库")
        except Exception as e:
            logger.error(f"加载自定义海龟汤失败: {e}")
            self.stories = []

    def save_stories(self):
        """保存自定义故事到文件"""
        try:
            storage_path = (
                self.storage_file
                if isinstance(self.storage_file, str)
                else str(self.storage_file)
            )
            # 确保目录存在
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
            with open(storage_path, "w", encoding="utf-8") as f:
                json.dump(self.stories, f, ensure_ascii=False, indent=2)
            logger.info(f"保存了 {len(self.stories)} 个自定义海龟汤故事到 {storage_path}")
        except Exception as e:
            logger.error(f"保存自定义海龟汤失败: {e}")

    def add_story(self, puzzle: str, answer: str) -> bool:
        """添加自定义故事到存储库"""
        with self.lock:
            story = {
                "puzzle": puzzle,
                "answer": answer,
                "created_at": datetime.now().isoformat(),
            }
            self.stories.append(story)
            self.save_stories()
            logger.info(f"添加新自定义海龟汤故事，当前存储库大小: {len(self.stories)}")
            return True

    def get_story(self) -> Optional[Tuple[str, str]]:
        """从自定义存储库获取一个故事，避免重复（线程安全）"""
        if not self.stories:
            return None

        with self.lock:
            # 获取所有可用的索引（排除已使用的）
            available_indexes = [
                i for i in range(len(self.stories)) if i not in self.used_indexes
            ]

            # 如果没有可用题目，清空已用记录，重新开始一轮
            if not available_indexes:
                logger.info("自定义存储库已全部使用完毕，清空记录重新开始")
                self.used_indexes.clear()
                available_indexes = list(range(len(self.stories)))
                # 立即保存重置后的状态
                self.save_usage_record()

            # 从可用索引中随机选择一个
            import random

            selected = random.choice(available_indexes)
            self.used_indexes.add(selected)

            # 保存使用记录
            self.save_usage_record()

            story = self.stories[selected]
            logger.info(
                f"从自定义存储库获取故事，索引: {selected}, 已使用: {len(self.used_indexes)}/{len(self.stories)}"
            )
            return story["puzzle"], story["answer"]

    def get_storage_info(self) -> Dict:
        """获取自定义存储库信息"""
        usage_info = self.get_usage_info()
        return {
            "total": len(self.stories),
            "used": usage_info["used"],
            "remaining": len(self.stories) - usage_info["used"],
        }


# 验证结果类
