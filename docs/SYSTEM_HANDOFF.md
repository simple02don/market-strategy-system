# A股主力策略情景推演与分层选股系统 · 接手文件

> 最后核验时间：2026-08-05 01:30
> GitHub（目标）：`simple02don/market-strategy-system`（私有；Codex GitHub App
>  尚需在该仓库授权 Contents 写权限后才能推送）
> 生产：`root@43.136.54.243:/home/ubuntu/market-strategy-system`（ubuntu 运行）

## 1. 系统定位

- 每日 23:00（仅“下一自然日为交易日”时）生成次日全景概率推演：市场状态 →
  次日情景 → 板块职责 → 个股 0-3 主推荐，并通过企业微信推送。
- 与 JCKX Tail Overnight 完全独立：不 import 其代码、不写其数据目录、不用其
  锁名/cron。只读复用其事件缓存与分钟档案，且必须带版本校验和直连兜底。
- 只读复用共享凭据：Tushare token、企业微信 webhook、JCKX 报告密码、DeepSeek key。
- 只输出研究推演，不自动下单，不构成投资建议。

## 2. 当前状态

| 项目 | 状态 |
|---|---|
| 部署 | `/home/ubuntu/market-strategy-system`，venv 独立，`.env` 600 |
| 调度 | 23:00 `nightly`（已开启推送）；23:03 health；23:10 track-outcomes；周六 02:00 train / 03:00 backtest |
| 数据 | Tushare 3 年回灌（725 交易日 / 390 万日线 / 390 万每日指标 / 指数），PIT 字段已建 |
| 新闻/事实 | 财联社(Tushare)/巨潮/政府网/部委多源；DeepSeek 原子事实抽取可用 |
| 模型 | HMM + LightGBM + Isotonic 校准 v2 已训练并上线（个股 RankIC 0.046，可执行宇宙 175 万行） |
| 回测 | 2026-08-05 首次回测（可执行宇宙、100 天样本外）：市场方向 Brier 0.324 暂劣于无条件基线；板块 RankIC 0.018；个股 RankIC 0.042 优于动量/反转基线；Top10 组合日均超额 -0.13%（成本后），动量基线 -0.84%。结论：个股排序有弱信号，市场方向与组合净值尚未跑赢成本，继续影子验证 |
| 推送 | 2026-08-05 01:27 首次真实推送成功（8/6 报告，11 只候选） |
| 报告服务 | `127.0.0.1:8082`，nginx `/strategy/` 反代，JCKX 密码登录，WireGuard 内网 |
| 原系统 | 未改动（nginx 仅新增 location，配置备份 `/root/nginx_jckx-reports.bak.20260804`） |

## 3. 常用命令

```bash
cd /home/ubuntu/market-strategy-system
./run.sh check-calendar        # 明天是否交易日、今晚是否运行
./run.sh data-update --trade-date YYYYMMDD
./run.sh nightly --no-push     # 生成报告不推送（自检）
./run.sh train                 # 模型训练（周六自动）
./run.sh health
```

## 4. 数据与模型

- SQLite：`data/market_strategy.sqlite3`（交易日历/日线/每日指标/新闻/事实/预测日志）。
- 模型产物：`models/artifacts/v{n}`（LightGBM txt + HMM/校准器 joblib + meta.json），
  23:00 只加载最新版本推理；冠军/挑战者机制：样本外指标不劣于现有才替换。
- 训练与推理分离：训练在低峰时段自动运行，不在 23:00 任务内训练。
- 时间一律 Asia/Shanghai（`timeutil.now_cst`），报告记录 decision_time / information_cutoff /
  dataset_version / model_version / code_commit。

## 5. 关键约束

- 免费数据源缺失/不兼容时必须回退自拉，共享只是优化不是依赖。
- 数据失败推“失败说明”，进入降级/弃权，不静默。
- 禁止为出票放低硬门槛（市值≥110亿、PE 0-300、主板+创业板、剔除 ST/科创板/
  北交所/停牌/一字板/上市不足60日）。
- 收假前夜运行必须纳入整个假期资讯窗口，并标注最近行情日陈旧天数。
- 不打印 `.env`/token/webhook 正文。

## 6. 待办

- GitHub 推送：Codex 桌面端重新连接 GitHub App 以签发含新仓库权限的 token，之后推送代码并启用 CI。
- 市场方向模型改进（当前 Brier 劣于无条件基线，暂不作为主决策依据）。
- 组合成本后净值跟踪：连续样本窗口验证个股 RankIC 是否能转化为可执行收益。
- 降级状态机（facts_only/abstain）细节完善与告警分级。
