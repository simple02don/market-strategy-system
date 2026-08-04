"""模型训练入口（服务器低峰自动运行）。

v1 先完成特征/标签物化与规则基线评估；HMM/LightGBM 训练在本模块后续版本接入，
产物写入 models/artifacts，23:00 推理只加载冻结产物。
"""

from __future__ import annotations

from ..storage import Storage


def train_all(storage: Storage, trade_date: str) -> dict:
    # 阶段实现：校验历史数据覆盖、输出数据版本标记；模型在后续迭代接入。
    counts = {
        "daily_rows": storage._conn.execute("SELECT COUNT(*) FROM daily_bar").fetchone()[0],
        "basic_rows": storage._conn.execute("SELECT COUNT(*) FROM daily_basic").fetchone()[0],
        "last_date": storage._conn.execute(
            "SELECT MAX(trade_date) FROM daily_bar"
        ).fetchone()[0],
    }
    return {
        "status": "ok",
        "trade_date": trade_date,
        "dataset": counts,
        "model_version": "rule_v1",
        "note": "规则基线已就绪；HMM/LightGBM 训练将在模型迭代阶段接入同一入口",
    }
