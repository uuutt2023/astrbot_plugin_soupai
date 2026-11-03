"""传输层错误处理的测试。

锁住三件事：
- 500 归入「上游故障」并参与重试（很多中转网关在上游失败时返回 500）；
- 错误体的字符串形态与顶层 message 也要能读出来，否则日志里只剩「HTTP 500」；
- 请求本身有问题的状态不该重试。
"""

import importlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))
transport = importlib.import_module(f"{_ROOT.name}.core.jev_transport")


class TestErrorStatus:
    def test_500_is_upstream_error(self):
        assert transport.error_status(500) == "upstream_error"

    def test_known_statuses(self):
        assert transport.error_status(401) == "unauthorized"
        assert transport.error_status(402) == "insufficient_credits"
        assert transport.error_status(429) == "rate_limited"
        assert transport.error_status(502) == "upstream_error"


class TestRetryable:
    def test_500_is_retryable(self):
        assert 500 in transport.RETRYABLE_STATUS
        assert 502 in transport.RETRYABLE_STATUS

    def test_client_errors_not_retryable(self):
        for status in (400, 401, 402, 403, 404, 422):
            assert status not in transport.RETRYABLE_STATUS


class TestParseError:
    def test_object_form(self):
        kind, message = transport.parse_error(
            {"error": {"type": "authentication_error", "message": "Invalid key"}}
        )
        assert kind == "authentication_error"
        assert message == "Invalid key"

    def test_plain_string_form(self):
        """中转网关常把 error 写成字符串：{"error": "Task request failed"}。"""
        kind, message = transport.parse_error({"error": "Task request failed"})
        assert kind == ""
        assert message == "Task request failed"

    def test_top_level_message(self):
        kind, message = transport.parse_error({"message": "invalid api key"})
        assert message == "invalid api key"

    def test_non_dict(self):
        assert transport.parse_error("boom") == ("", "")
