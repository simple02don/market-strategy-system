# A股主力策略情景推演与分层选股系统 · 接手文件

> 最后核验时间：2026-08-06 03:40 CST
> 本文件是“接手入口”，详细规则见 [AGENTS.md](../AGENTS.md)，功能与验收分期见
> [REPAIR_PHASES.md](REPAIR_PHASES.md)，定位与用法见 [README.md](../README.md)。
> 敏感信息（token/密码/webhook/key）一律不写入本文件；真实值只在生产 `.env`。

## 1. 当前状态速览

| 项 | 状态 |
|---|---|
| 代码 | 本地 HEAD `7ebd4e3`，GitHub `simple02don/market-strategy-system` main 同步；生产部署提交 `e118d88`（tar 部署，生产 git HEAD 与代码一致） |
| 测试 | 38 个全过（本地 Python 3.10 venv + 生产 Python 3.12 venv） |
| 生产 | `/home/ubuntu/market-strategy-system`（ubuntu 运行，`.env` 权限 600），`root@43.136.54.243`（ssh 别名 `jckx-prod`，密钥 `~/.ssh/jckx_prod_ed25519`） |
| 数据 | SQLite `data/market_strategy.sqlite3`，最新行情日 20260805（2026-08-06 晚间将自动补 20260806） |
| 模型 | `models/artifacts/v1`、`v2`（gitignored）；v2 组件未获批准 → 线上走规则基线 `rule_v1` |
| 推送 | 企业微信 webhook 正常；8/7 报告已推送（run 21，8/5 数据版）；今晚 23:00 会用 8/6 数据再推一次（数据新鲜度防重保证不会被跳过） |
| 报告访问 | 公网 HTTPS `https://43.136.54.243/strategy/`（登录密码 + 限速），WireGuard 内网 `http://10.66.0.1/strategy/` 仍可用 |
| 证书 | Let's Encrypt 短效 IP 证书已续期（8/12 到期），每 6 小时自动续期并 reload nginx |

## 2. 常用命令

本地（`PYTHONPATH=src`）：
```bash
PYTHONPATH=src ./venv/bin/python -m pytest tests/ -q
PYTHONPATH=src ./venv/bin/python -m market_strategy.cli check-calendar
```

生产（`run.sh` 已自动加载 `.env` 并用 ubuntu 身份执行）：
```bash
ssh jckx-prod
cd /home/ubuntu/market-strategy-system
./run.sh check-calendar        # 今晚是否运行
./run.sh nightly --no-push     # 干跑（不推送，不写正式预测）
./run.sh nightly --force       # 强制重跑（绕过防重，正式推送）
./run.sh health                # 健康检查（23:08 自动跑）
./run.sh track-outcomes        # 结果跟踪 + 分钟级回放（23:15 自动跑）
./run.sh train                 # 周六 02:00 自动训练
./run.sh train-log             # 训练实验记录（P1-5 新增）
./run.sh backtest              # 周六 03:00 自动回测
```

## 3. 架构与数据

主链路（23:00，仅“下一自然日为交易日”时）：
```
日历判定 → 数据更新（日线/指标/指数/龙虎榜，Tushare 为主，
           指数回退腾讯，龙虎榜失败只降级不阻塞）
→ 新闻/政策/公告多源采集 → PIT 过滤与跨源去重
→ DeepSeek 有界影响评估（词典降级 + 截断重试）
→ 证据融合（含龙虎榜资金面）→ 市场状态 → 次日情景 → 板块 → 个股
→ 主推荐行业分散（同行业≤2）→ HTML 报告 → 企业微信推送（公网链接）
```

核心表（`Storage` 自动建表/迁移）：
`trade_cal / stock_basic / daily_bar / daily_basic / index_daily /
lhb_daily / lhb_inst / news_item / news_impact / atomic_fact /
run_log / prediction_log / evidence_snapshot / candidate_outcome /
execution_replay / minute_bar / train_experiment`

模型产物：`models/artifacts/v{n}`，训练写临时目录后原子发布；冠军/挑战者机制，
组件级批准（`meta.component_status`），未批准组件线上回退规则。

## 4. 已实现功能（按最近工作）

- 资讯成为决策证据：严格 `information_cutoff`、跨源去重、LLM 有界影响评估、
  词典兜底、`evidence_snapshot` 存档（阶段 1）
- 线上正确性与降级：`facts_only` / `abstain` 取消主推荐；硬门槛
  （市值/PE/上市天数/成交额/ST/科创板/北交所/一字板）；NaN 清理（阶段 2）
- 训练/回测/晋级：四段切分、每日横截面 RankIC、成本后 Top-K、组件批准、
  原子发布与回退（阶段 3）
- 数据日解析按“数据可得时点”（18:00 前用前一交易日），防重按数据新鲜度比较，
  凌晨手动运行不会挡今晚 23:00 正式运行
- 龙虎榜资金面：`top_list`/`top_inst` 入库、按行业聚合、进入操盘假设与报告
- 操盘假设对抗化：每个假设带支持/反证/最强反证/为何未采纳/次日验证
- 主推荐行业分散：`PRIMARY_MAX_SAME_INDUSTRY=2`（规则与模型两条路径都生效）
- 指数回退源：Tushare → 腾讯日 K（东财会拦截 python requests，已弃用）
- 分钟级回放：Tushare `stk_mins`（主）→ 新浪（当日）→ 东财 curl（兜底）；
  23:15 对到期候选回放“高开≤3% 且 15 分钟站稳分时均线”等确认/取消条件
- 训练实验记录：`train_experiment` 表 + `train-log`；市场模型 walk-forward
  稳定性指标（8 折，暂只参考不进晋级门槛）
- 公网报告：nginx 443 HTTPS（仅 `/strategy/`，登录+限速，Secure cookie），
  80 口 ACME 验证；WireGuard 路径不变，JCKX 原报告仍仅内网

## 5. 生产环境要点

- `.env` 需要的键（值只在生产）：`TUSHARE_TOKEN / WECOM_WEBHOOK /
  JCKX_PASSWORD / JCKX_TOKEN / AI_PRIMARY_API_KEY / AI_PRIMARY_BASE_URL /
  AI_PRIMARY_MODEL / SHARED_EVENT_CACHE_DIR / SHARED_MINUTE_HISTORY_DIR /
  PRIMARY_MAX / PRIMARY_MAX_SAME_INDUSTRY / WATCH_MAX / MIN_* / NLP_* /
  JCKX_REPORT_BASE_URL` 等，完整见 `.env.example`
- 共享只读 JCKX 缓存：`/home/ubuntu/jckx-tail-overnight/cache`（事件 pickle，
  带版本校验）；`data/minute_history` 只有指数分钟，无个股分钟
- nginx：`0.0.0.0:443`（公网 `/strategy/`）、`0.0.0.0:80`（ACME+跳转）、
  WireGuard `10.66.0.1:80`（JCKX 8081 + strategy 8082）；ufw 放行 22/80/443/51820
- 云安全组：22、80、443、51820/udp 已放行
- auth_server：`127.0.0.1:8082`，`logs/auth_server.pid` + 每 5 分钟看门狗
- cron（ubuntu）：23:00 nightly（推送）、23:08 health、23:15 track-outcomes、
  周六 02:00 train、周六 03:00 backtest、auth 看门狗

## 6. 在新电脑继续开发

```bash
git clone git@github.com:simple02don/market-strategy-system.git
cd market-strategy-system
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
cp .env.example .env          # 从生产抄真实值：ssh jckx-prod 'cat /home/ubuntu/market-strategy-system/.env'
PYTHONPATH=src ./venv/bin/python -m pytest tests/ -q   # 期望 38 passed
```

注意：
- macOS（Intel）上 lightgbm 需要 OpenMP：把 `libomp.dylib` 放到
  `/usr/local/opt/libomp/lib/libomp.dylib`（可从 conda-forge osx-64
  `llvm-openmp` 包提取，或用 Homebrew bottle；本机已装）
- 本地没有生产数据时，可 `scp` 生产 `data/market_strategy.sqlite3` 回来做只读
  分析，或 `./run.sh data-backfill --years 1` 重新拉（慢）
- GitHub 推送用 `GIT_SSH_COMMAND="ssh -i ~/.ssh/github_simple02don_ed25519 -o IdentitiesOnly=yes"`

## 7. 发布/部署流程（当前实践）

```bash
# 本地
PYTHONPATH=src ./venv/bin/python -m pytest tests/ -q
git add -A && git commit -m "..." && git push origin main
git diff --name-only HEAD~1 | tar -czf /tmp/mss.tar.gz -T -
scp /tmp/mss.tar.gz jckx-prod:/tmp/
# 生产
ssh jckx-prod 'runuser -u ubuntu -- sh -c "cd /home/ubuntu/market-strategy-system && tar -xzf /tmp/mss.tar.gz && PYTHONPATH=src ./venv/bin/python -m pytest tests/ -q && git add -A && git -c user.name=deploy -c user.email=deploy@local commit -m \"deploy: ...\" "'
# 若改了 auth_server.py：重启
ssh jckx-prod 'runuser -u ubuntu -- sh -c "pkill -u ubuntu -f auth_server.py; sleep 1; rm -f /home/ubuntu/market-strategy-system/logs/auth_server.pid; cd /home/ubuntu/market-strategy-system && ./start_http.sh"'
```

验收顺序（来自 REPAIR_PHASES.md）：隔离目录全量测试 → 生产测试 →
`nightly --no-push` 干跑 → 核查报告（system_status/证据/无 NaN）→ 再恢复正式推送。

## 8. 已知边界与待办

- 资讯增量 A/B（有新闻 vs 无新闻）需 60 个正式交易日数据；分钟回放首条结果
  将在 2026-08-07 收盘后产生
- 历史 ST 名称/行业变更仍非完整 PIT；行情修订历史未版本化
- walk-forward 指标暂不进晋级门槛，观察几周后考虑
- 个股日线的多源回退未做（当前 Tushare 重试 + 库内降级）
- 18:00 数据可得阈值是启发式；凌晨边界运行可能走降级（安全）
- 共享缓存事件 source_id 临时化，可能重复送 LLM 评估（成本提示）

## 9. 建议技能（给接手 Agent）

- `diagnosing-bugs`：系统运行异常时先走诊断循环
- `tdd`：新增/修复功能时测试先行（仓库已有较完整回归测试）
- `self-improvement`：出现意外失败或纠正时记录教训
- `handoff`：本会话继续交接时重新生成交接文档

## 10. 近期提交（可作改动依据）

`7ebd4e3` walk-forward；`d4b7ccd` 训练实验记录；`2546946` 分钟回放；
`d322c01`/`bf45c1f` 指数回退；`1543595` 龙虎榜证据；`57f1071` 公网认证加固；
`b51980a` 行业分散；`44f5c25` 数据新鲜度防重；`5b2993c`/`445e996` 覆盖率修复。
