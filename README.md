# A股主力策略情景推演与分层选股系统

独立于 JCKX Tail Overnight 的新系统：每日 23:00 生成“下一交易日”的全景概率
推演（市场状态 → 次日情景 → 板块职责 → 个股推荐），并通过企业微信推送给用户。

## 核心规则

- 23:00 只在“明天是交易日”时运行并推送；周五、假期中不推，收假前夜推一次且
  纳入整个假期的资讯。
- 当前链路：严格信息截止过滤 → 跨源去重 → DeepSeek 有界影响评估（词典降级）→
  市场状态/次日情景/板块/个股证据融合 → 0-3 只主推荐。
- “主力/操盘行为”只作为竞争性假设输出，同时列出支持证据、反证和次日验证条件，
  不把不可观测主体写成确定事实。
- 原系统（jckx-tail-overnight）不动：不 import 其代码、不写其数据目录、不用其
  锁名与 cron。仅允许只读复用其事件缓存与分钟档案，且必须带版本校验和直连兜底。
- 共享：同一台服务器、同一 Tushare token、同一企业微信 webhook、同一 JCKX 报告
  密码；报告只走 WireGuard 内网。
- 新闻/政策/公告多免费源接入：财联社电报、巨潮公告、中国政府网与部委政策库；
  跨源标题去重、每源覆盖监控，资讯或 LLM 影响评估不足时进入 `facts_only` 并取消
  主推荐。重大政策和公告额外做原文事实抽取与原文片段核验。

## 运行方式

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env   # 填入 Tushare token、企业微信 webhook、AI key

./run.sh check-calendar          # 明天是否交易日、今晚是否推送
./run.sh data-backfill --years 3 # 首次历史数据回灌（耗时较长）
./run.sh data-update             # 增量更新当日数据
./run.sh nightly --no-push       # 生成今晚报告但不推送（自检）
./run.sh nightly                 # 正式夜间运行并推送
./run.sh train                   # 模型训练（低峰时段，全自动）
./run.sh health                  # 健康检查
```

## 调度（独立 cron）

- 每天 23:00：`nightly`（明天不是交易日则静默退出）
- 每天 23:08：`health`（核验夜间运行、资讯证据状态与推送）
- 每周六 02:00：`train`（模型重训，不阻塞 23:00 推理）
- 报告服务：`@reboot` + 每 5 分钟 watchdog，监听 `127.0.0.1:8082`，nginx
  `/strategy/` 反代，仅 WireGuard 内网可访问。

## 目录

```text
src/market_strategy/
├── calendar.py        # 交易日历与“明天是否交易日”判定
├── storage.py         # SQLite：交易日历/日线/每日指标/事实/预测日志
├── providers/         # Tushare 直连、原系统共享缓存只读适配
├── features/          # 市场宽度/板块/个股特征
├── models/            # 状态识别/转移/情景/板块/个股 + 校准
├── nlp/               # 政策/公告原文事实抽取
├── pipeline.py        # 23:00 编排
├── report.py          # HTML 日报
└── push/wecom.py      # 企业微信摘要推送
```

本系统只生成研究与概率推演，不自动下单，不构成投资建议。
