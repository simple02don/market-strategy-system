# market-strategy-system 操作规则

## 硬约束

- 本仓库与 `jckx-tail-overnight` 完全独立。禁止 import 原系统代码、禁止写原系统
  任何数据目录、禁止使用原系统锁名或 cron。
- 只读复用原系统缓存（事件 pickle、分钟档案）必须带格式/版本校验；缺失或不兼容
  时必须回退到本系统自己拉取，共享是优化不是依赖。
- 23:00 只在“下一自然日为交易日”时运行并推送；收假前夜那次运行必须纳入整个
  假期的资讯窗口，并显著标注最近行情日。
- 不输出 `.env`、token、webhook、私钥正文。
- 训练与推理分离：23:00 任务只推理；训练在低峰时段（每周六 02:00）自动运行。
- 模型晋级必须样本外含成本稳定优于基线；禁止为出票放低硬门槛。
- 数据失败时推送失败说明并进入 `facts_only`/`abstain`，不得静默。
- 新闻/政策/公告以免费一手/二手源为主（财联社、东财、巨潮、政府网），必须做跨源
  去重、来源分级、原文归档与每源覆盖度监控；AI 抽取只用于高价值文档。

## 部署拓扑

- GitHub（目标）：`simple02don/market-strategy-system`（私有）。
- 生产：`/home/ubuntu/market-strategy-system`（ubuntu），独立 venv、独立 `.env`
  （600）、独立锁 `/tmp/market_strategy_*.lock`、独立 cron。
- 报告：`127.0.0.1:8082` + nginx `location /strategy/`，仅 WireGuard 内网。
- 发布：本地提交 → GitHub → 生产 fetch + fast-forward → 完整测试 → 验收。

## 数据时点纪律

- 每条预测保存 `decision_time`、`information_cutoff`、`dataset_version`、
  `model_version`、`code_commit`。
- 股票池/行业分类/指数成分按历史日期重建；训练与在线推理用同一 PIT 规则。
