"""进行中的对局状态：群聊 ID → 本局游戏数据。

只放内存，不落盘——插件重载后对局自然作废（原行为如此）。
"""

from typing import Dict, Optional


class GameState:
    def __init__(self):
        self.active_games: Dict[str, Dict] = {}  # 群聊ID -> 游戏状态

    def start_game(self, group_id: str, puzzle: str, answer: str, **extra) -> bool:
        """开始游戏，返回是否成功"""
        if group_id in self.active_games:
            return False
        game_data = {
            "puzzle": puzzle,
            "answer": answer,
            "is_active": True,
            "qa_history": [],
            "hint_history": [],
        }
        game_data.update(extra)
        self.active_games[group_id] = game_data
        return True

    def end_game(self, group_id: str) -> bool:
        """结束游戏"""
        if group_id in self.active_games:
            del self.active_games[group_id]
            return True
        return False

    def get_game(self, group_id: str) -> Optional[Dict]:
        """获取游戏状态"""
        return self.active_games.get(group_id)

    def is_game_active(self, group_id: str) -> bool:
        """检查是否有活跃游戏"""
        return group_id in self.active_games


# 网络海龟汤管理
