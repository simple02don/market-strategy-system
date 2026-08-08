from datetime import datetime

from market_strategy.cli import _tail_review_required


def test_tail_review_required_when_latest_data_is_today_even_if_tomorrow_is_closed():
    current = datetime(2026, 8, 7, 23, 8)

    assert _tail_review_required("20260807", current) is True
    assert _tail_review_required("20260806", current) is False
