# A股主力策略情景推演与分层选股系统 · 接手文件

> 最后核验时间：2026-08-07（周五）
> 详细规则见 [AGENTS.md](../AGENTS.md)，功能分期见 [REPAIR_PHASES.md](REPAIR_PHASES.md)，
> 定位与用法见 [README.md](../README.md)。本文件只写键名，不写任何 token/密码。

## 1. 当前状态速览

| 项 | 状态 |
|---|---|
| 代码 | GitHub PR #1 已合并（业务合并提交 `ff0fa16`）；生产部署提交 `eafdbcf`，业务文件与 GitHub 合并内容一致 |
| 测试 | 81 个全过（隔离目录与生产目录各一次，Python 3.12）；compileall 通过 |
| 生产 | `/home/ubuntu/market-strategy-system`（ubuntu 运行，`.env` 600）；`ssh jckx-prod`（root@43.136.54.243，密钥 `~/.ssh/jckx_prod_ed25519`） |
| 数据 | SQLite `data/market_strategy.sqlite3`，最新行情日 20260807；部署前在线备份位于 `/home/ubuntu/market-strategy-deploy-backups/pre-e22553b/` |
| 模型 | `models/artifacts/v1`、`v2`（gitignored）；组件未批准 → 线上走规则基线 |
| 推送 | 最近正式推送仍为 run 35；部署验收 run 36 使用 8/7 数据生成 8/10 干跑报告，`normal`、无正式记录、未推送；周日 23:00 按日历生成正式版 |
| 报告访问 | 公网 HTTPS：市场策略 `/strategy/`、JCKX 原报告 `/jckx/`（均登录，会话 10 年）；WireGuard 内网路径不变 |
| 证书 | Let's Encrypt 短效 IP 证书，每 6 小时自动续期并 reload nginx |

## 2. 常用命令

本地：
```bash
PYTHONPATH=src ./venv/bin/python -m pytest tests/ -q
PYTHONPATH=src ./venv/bin/python -m market_strategy.cli check-calendar
```

生产（run.sh 自动加载 .env、以 ubuntu 身份运行）：
```bash
cd /home/ubuntu/market-strategy-system
./run.sh check-calendar
./run.sh nightly --no-push                  # 干跑
./run.sh nightly --force                    # 强制重跑并推送
./run.sh nightly --trade-date 20260810 --force --no-push   # 指定目标日干跑
./run.sh health                             # 23:08 自动跑
./run.sh track-outcomes                     # 23:15 自动跑（结果跟踪+分钟回放）
./run.sh train / train-log / backtest       # 周六自动
```

## 3. 主链路（当前版本）

```
日历判定 → 数据更新（Tushare 主，指数回退腾讯；LHB 入库失败只降级）
→ 多源新闻 → PIT 过滤/去重 → DeepSeek 有界影响评估（词典兜底+截断重试）
→ 证据融合（含龙虎榜、行业标签规范化）
→ 当日焦点板块（今日涨幅+涨停效应+量能，过滤小行业）
→ 主力阶段机：吸筹/洗盘/拉升/拉升高潮/派发/砸盘/反包/观望
  （看收盘位置、上下影、量价组合、板块分化，输出恶意证据 trap_signals）
→ 阶段转移预判下一交易日（收割剧本）→ 目标板块
→ 三形态选股（刚启动/可控回踩/上升趋势）+ 支撑/压力位 + near-miss 兜底
→ 防守模式（预判派发/砸盘/观望时）：反包猎手/超跌修复/避风港轮动
→ HTML 报告（意图推演/阶段手册/防守机会/支撑压力列）→ 企业微信推送
```

## 4. 已实现功能（按模块）

- **证据层**：严格 information_cutoff、官方列表日期提取、跨源去重、LLM 有界影响评估、词典兜底、
  行业标签规范化（SECTOR_TERMS + SECTOR_ALIASES + known_industries）、
  科创板代码排除、正文快照/source hash、evidence_snapshot 存档
- **龙虎榜**：top_list/top_inst 入库，同行业正负个股资金先净额相抵并记录广度，
  再进入操盘假设与防守选股
- **主力阶段机**（`models/intent.py`）：
  - 每日意图=阶段判定（吸筹/洗盘/拉升/拉升高潮/派发/砸盘/反包/观望），
    输出 trap_signals（收割信号）
  - 阶段转移规则：拉升+追高（涨停潮+放量+上影+连续）→派发→砸盘；
    砸盘长下影收回→反包；反包缩量弱→诱多再砸
  - 每阶段应对手册（STAGE_PLAYBOOK：操作/战术/风险）
- **选股**（`models/stock_pattern.py`）：
  - 三形态：刚启动（放量突破20日平台）/可控回踩/上升趋势
  - 支撑压力：20日前低、20/60日前高、上方空间、距支撑距离
  - near-miss 兜底：正常模式 0 合格主推时，用“上升雏形/突破量能略欠/
    回踩略超窗口”补位（仅无合格时启用）
  - 防守模式：反包猎手（被砸板块未破位+止跌特征）、超跌修复、
    避风港轮动（行业净流入+资金广度+个股 MA20 结构+直接资金方向，≤15%仓位）
- **硬过滤**：常规/防守池共用 ST、板块、市值、PE、流动性、上市天数、涨停和过热过滤；
  目标板块候选在形态筛选前不被全市场榜单截断
- **训练/实验**：四段切分、每日截面 RankIC、成本后回测、滚动/逐日稳定性门槛、组件批准、
  原子发布+回退、train_experiment 落库、train-log、市场 walk-forward
- **回放**：Tushare 分钟主源+新浪+东财兜底；主推荐/反包/修复/避风港各自冻结
  机器可读执行计划，完整 15 根确认、确认价成交、no_data 跨天重试；观察/回避层
  不进入成交与收益统计，旧开盘代理样本从新指标隔离
- **报告/推送**：主力意图推演、恶意证据、阶段应对手册、防守机会、
  支撑/压力列；明确数据截止日/预测目标日/运行模式，推送成功后才记正式；
  企业微信含预判/防守机会行；公网链接

## 5. 生产环境要点

- `.env` 键：`TUSHARE_TOKEN / WECOM_WEBHOOK / JCKX_PASSWORD / JCKX_TOKEN /
  AI_PRIMARY_API_KEY / AI_PRIMARY_BASE_URL / AI_PRIMARY_MODEL /
  SHARED_EVENT_CACHE_DIR / SHARED_EVENT_CACHE_MAX_BYTES / SHARED_MINUTE_HISTORY_DIR / PRIMARY_MAX /
  PRIMARY_MAX_SAME_INDUSTRY / WATCH_MAX / MIN_* / NLP_* / JCKX_REPORT_BASE_URL` 等
- nginx：`0.0.0.0:443`（`/strategy/` → 8082，`/jckx/` → 8081，均登录+限速；
  `/jckx/` 带 proxy_redirect + sub_filter 修正登录表单）；
  `0.0.0.0:80`（ACME+跳转）；WireGuard `10.66.0.1:80`（JCKX 根路径仅内网）
- 会话：10 年（`JCKX_SESSION_MAX_AGE=315360000`），SameSite=Lax，Secure（HTTPS）
- cron（ubuntu）：23:00 nightly、23:08 health、23:15 track-outcomes、
  周六 02:00 train、周六 03:00 backtest、auth 看门狗每 5 分钟
- 证书续期：cron `17 */6 * * *` certbot renew + reload nginx；webroot 公网 80

## 6. 在新电脑继续开发

```bash
git clone git@github.com:simple02don/market-strategy-system.git
cd market-strategy-system
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
cp .env.example .env   # 真实值从生产抄：ssh jckx-prod 'cat .../.env'
PYTHONPATH=src ./venv/bin/python -m pytest tests/ -q   # 期望 81 passed
```

注意：macOS Intel 上 lightgbm 需要 libomp（放 `/usr/local/opt/libomp/lib/libomp.dylib`，
可从 conda-forge osx-64 llvm-openmp 包提取）；本地无生产数据时可 scp 数据库或
`./run.sh data-backfill`；推送用 `GIT_SSH_COMMAND="ssh -i ~/.ssh/github_simple02don_ed25519 ..."`。

## 7. 发布/部署流程

```bash
# 本地：测试 → 提交 → 推送
PYTHONPATH=src ./venv/bin/python -m pytest tests/ -q
git add -A && git commit -m "..." && git push origin main
git diff --name-only HEAD~1 | tar -czf /tmp/mss.tar.gz -T -
scp /tmp/mss.tar.gz jckx-prod:/tmp/
# 生产：解压 → 测试 → 提交部署 commit
ssh jckx-prod 'runuser -u ubuntu -- sh -c "cd ... && tar -xzf /tmp/mss.tar.gz \
  && PYTHONPATH=src ./venv/bin/python -m pytest tests/ -q \
  && git add -A && git -c user.name=deploy -c user.email=deploy@local commit -m \"deploy: ...\" "'
# 若改了 auth_server.py：重启对应服务（见第 5 节）
```

验收顺序（REPAIR_PHASES.md）：隔离目录全量测试 → 生产测试 → `--no-push` 干跑
→ 核查报告（normal/证据/无 NaN）→ 再恢复正式推送。

## 8. 已知边界与待办

- **报告页面圈注修改待办**：用户提供了一张带圈注的报告截图要求改版，
  但当前会话无法读取图片，已请用户用文字描述；拿到意见后严格只改圈注处，
  未圈处只做色彩/明暗调整（改 `report.py`，生成后用浏览器核对）。
- 阶段机与三形态阈值目前是启发式；需积累 2-4 周 outcome+回放数据后回标定。
- 正常模式仍有极小概率 0 主推：目标板块内连 near-miss 都不合格时（设计如此）。
- 历史 ST/行业 PIT 不完整；行情修订历史未版本化。
- 资讯增量 A/B（有新闻 vs 无新闻）待 60 个正式交易日数据。
- JCKX 系统（另一仓库 `jckx-tail-overnight`）白天报告 0 推荐是设计内保守行为；
  已放宽 `TAIL_AI_WATCH_UPGRADE_MIN_SCORE=60`、`TAIL_OPPORTUNITY_MIN_AI_SCORE=50`，
  其推送链接已改公网 `/jckx/`。
- 共享缓存只有指数分钟，无个股分钟；分钟回放依赖 Tushare stk_mins。

## 9. 建议技能（给接手 Agent）

- `diagnosing-bugs`：运行异常先走诊断循环
- `tdd`：改代码测试先行（仓库已有 81 个回归测试）
- `self-improvement`：意外失败或纠正时记录教训
- `handoff`：再次交接时重新生成本文件

## 10. 近期提交（改动依据）

`507cfb4` 统一硬过滤/收益与分钟回放正确性；`e22553b` 防守资金广度、结构资格、
机器可读执行计划与报告目标日语义；GitHub PR #1 合并提交 `ff0fa16`；生产部署提交
`eafdbcf`。其余历史依据：

`9edb19c` 防守候选形态保留；`e8f1810` 支撑压力+near-miss；`c11dcac`/`389b2dc`/
`5155f13`/`55b8f47`/`17ea830`/`86b2a0f` 防守机会与避风港；`a7c4cae`/`60f4f34`/
`67909f1`/`bb68056` 阶段机与收割剧本；`1543595` 龙虎榜证据；`57f1071` 公网加固；
`b51980a` 行业分散；`44f5c25` 数据新鲜度防重。
