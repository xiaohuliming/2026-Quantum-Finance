---
title: "QF-OPLRL Codex Plan: OPL 标准化、技术指标状态、QPL Gate V2"
author: "2026 Quantum Finance Project"
date: "2026-05-20"
geometry: margin=0.75in
fontsize: 10pt
header-includes:
  - \usepackage{fvextra}
  - \fvset{breaklines=true,breakanywhere=true}
---

# QF-OPLRL Codex Plan: 基于参考项目补齐当前代码缺口

## 0. 本次任务目标

当前项目已经具备数据加载、基础回测、传统 baseline、OPL baseline、Plain PPO、QPL 特征、QPL Gate、QPL Reward 和 ablation 脚本。下一步不是重写整个项目，而是在现有代码上做**最小侵入式升级**，补齐三个主要缺口：

1. **OPL 标准化缺口**：当前 `qf_oplrl/opl_baselines.py` 已有 BCRP、PAMR、OLMAR、ONS Diagonal，但需要参考经典 OLPS 项目核对公式、参数、输入输出、无未来信息泄漏和实验可比性。
2. **技术指标状态缺口**：PDF 愿景中的状态空间包含 `X_price + X_tech + X_qpl + w_{t-1}`，当前 RL 环境主要是 returns、previous weights 和 QPL features，缺少 MA、RSI、MACD、Volatility 等普通金融技术指标状态。
3. **QPL 风险门控缺口**：当前 `qpl_gate.py` 是离散 multiplier gate，逻辑过粗。需要升级为 QPL Gate V2，使其支持连续距离、动量、波动率、回撤、RL 加仓/减仓意图，以及可选 high/low 触碰 QPL 的逻辑。

最终目标：让项目从“能跑的课程项目框架”升级为“更规范、更贴近 PDF 愿景、更容易写报告和消融实验的 QF-OPLRL 实验系统”。

---

## 1. 先把参考 GitHub 项目克隆到 reference 文件夹

在 Windows PowerShell 或 Git Bash 中执行：

```bash
cd "C:\Users\27139\Desktop\Huge Workplace\2026 Quantum Finance\project reference"

git clone git@github.com:Marigold/universal-portfolios.git

git clone git@github.com:JarvisLee0423/Chaotic-Quantum-Finance-Trading-System.git
```

如果目录已经存在，改用更新命令：

```bash
git -C "C:\Users\27139\Desktop\Huge Workplace\2026 Quantum Finance\project reference\universal-portfolios" pull

git -C "C:\Users\27139\Desktop\Huge Workplace\2026 Quantum Finance\project reference\Chaotic-Quantum-Finance-Trading-System" pull
```

建议最终 reference 文件夹结构如下：

```text
C:\Users\27139\Desktop\Huge Workplace\2026 Quantum Finance\project reference
├── universal-portfolios
├── Chaotic-Quantum-Finance-Trading-System
├── FinRL
├── MILLION
├── OLPS
└── 其他已下载参考项目
```

---

## 2. 当前项目路径和关键文件

当前主项目路径：

```text
C:\Users\27139\Desktop\Huge Workplace\2026 Quantum Finance\project
```

Codex 需要重点检查和修改的文件：

```text
qf_oplrl/opl_baselines.py          # OPL baseline: BCRP, PAMR, OLMAR, ONS
qf_oplrl/plain_rl_env.py           # Plain RL environment
qf_oplrl/qpl_rl_env.py             # QPL-enhanced RL environment
qf_oplrl/qpl.py                    # QPL levels and QPL features
qf_oplrl/qpl_gate.py               # 当前 Gate V1
qf_oplrl/qpl_strategy.py           # QPL rule baseline and multiplier logic
qf_oplrl/data_loader.py            # 当前只抽取 close price matrix，可选补 OHLC
qf_oplrl/metrics.py                # 指标计算，可选补 regret
qf_oplrl/plots.py                  # 可选补图
scripts/run_opl_baselines.py       # OPL 实验入口
scripts/run_plain_rl.py            # Plain PPO 实验入口
scripts/run_qpl_ablation.py        # QPL ablation 实验入口
scripts/summarize_results.py       # 汇总结果入口
configs/dow30.yaml
configs/nas100.yaml
configs/olps.yaml
```

新增文件建议：

```text
qf_oplrl/technical_indicators.py   # 新增: MA, RSI, MACD, volatility
qf_oplrl/qpl_gate_v2.py            # 新增: 连续型、action-aware、risk-aware QPL Gate
scripts/run_qpl_gate_v2_ablation.py # 可选新增: 单独验证 Gate V1 vs Gate V2
```

---

## 3. 参考项目分工

### 3.1 universal-portfolios：用于 OPL 标准化

重点参考：

```text
project reference/universal-portfolios/universal/algos/
project reference/universal-portfolios/universal/algo.py
project reference/universal-portfolios/universal/result.py
project reference/universal-portfolios/examples/
```

Codex 需要用它核对：

- BCRP 是否作为 hindsight upper bound 使用；
- PAMR 的 loss、tau、simplex projection 是否标准；
- OLMAR 的 moving-average prediction 是否无未来信息泄漏；
- ONS 的实现是否过度简化；
- price relatives 输入格式是否与本项目一致；
- weights 的时间对齐是否满足“第 t 日权重只能使用 t-1 及以前信息”。

注意：不要把 `universal-portfolios` 整包硬复制进项目。只参考公式、参数和测试思想。当前项目应保持轻量实现。

### 3.2 Chaotic-Quantum-Finance-Trading-System：用于 QPL + Fuzzy Gate 思路

重点参考：

```text
project reference/Chaotic-Quantum-Finance-Trading-System/README.md
project reference/Chaotic-Quantum-Finance-Trading-System/FLStrategy.mq4
project reference/Chaotic-Quantum-Finance-Trading-System/Params.txt
```

Codex 需要参考它的思想，而不是照搬 MQL4 代码：

- QPL 不只是技术指标，而是支撑/阻力能级；
- Fuzzy logic 用来解决固定阈值导致的滞后和生硬触发问题；
- predicted high/low 或 observed high/low 可以用来判断是否触碰 QPL；
- Gate 输出不应该只有少数离散规则，最好允许连续 risk multiplier。

### 3.3 FinRL：用于技术指标状态和 portfolio RL environment 设计

重点参考：

```text
project reference/FinRL
```

Codex 只需要参考：

- portfolio allocation environment 的 state/action 结构；
- 技术指标作为 state features 的组织方式；
- action 用 softmax 转成 long-only portfolio weights 的思路；
- 不要引入 FinRL 的重型依赖，不要把项目改成 FinRL 框架。

### 3.4 MILLION：用于 risk-aware 实验包装

重点参考：

```text
project reference/MILLION
```

Codex 只需要参考它的 risk-aware 实验表达：

- 不只看 cumulative return；
- 强调 MDD、turnover、volatility、Calmar、Sharpe；
- 把 gate 包装成 risk control module，而不是单纯收益增强模块。

---

## 4. 任务 A：标准化 OPL baseline

### A1. 检查当前 `opl_baselines.py`

当前已有：

```text
project_to_simplex
bcrp
pamr
olmar
ons_diagonal
generate_opl_weights
```

Codex 需要逐一与 `universal-portfolios` 对照，重点检查：

1. **PAMR**
   - 当前公式是否为 mean-reversion 更新；
   - loss 是否应使用 `max(0, b_t^T x_t - epsilon)`；
   - tau 是否应支持 PAMR-0 / PAMR-1 / PAMR-2 变体；
   - `C` 参数是否与 config 一致。

2. **OLMAR**
   - moving average prediction 是否使用过去窗口；
   - 当前 `prices = price_relatives.cumprod()` 是否合理；
   - 第 t 日权重是否没有用到第 t 日之后价格；
   - window 和 epsilon 是否进入 config。

3. **ONS**
   - 当前 `ons_diagonal` 是简化 diagonal 版；
   - 需要在方法名中明确写成 `ONS Diagonal`，不要伪装成完整 ONS；
   - 可选实现 `ons_full`，但不强制。如果实现完整 ONS，必须保证复杂度在 DOW30 / NAS100 上可接受。

4. **BCRP**
   - BCRP 是 hindsight upper bound，只能用于测试集上界，不可作为可部署策略；
   - `run_opl_baselines.py` 当前在 `test_relatives` 上算 BCRP，这个方向是对的；
   - 需要在输出 CSV 里标记 `Method Type = Hindsight Upper Bound`。

### A2. 增加 OPL 验证测试

新增或补充一个轻量测试文件：

```text
tests/test_opl_baselines.py
```

最低要求：

- `project_to_simplex` 输出非负且和为 1；
- 每个 OPL 方法输出权重矩阵 shape 与输入 price relatives 一致；
- 每一行权重非负且和为 1；
- 所有方法在一个 3-asset toy dataset 上能跑完；
- BCRP wealth 不应低于 equal-weight CRP 太多；
- PAMR / OLMAR / ONS 不应产生 NaN/inf。

可选要求：

- 用 `universal-portfolios` 的同一 toy dataset 比较最终 wealth，允许一定误差；
- 如果 full ONS 没实现，测试中明确只验证 diagonal ONS。

### A3. 输出更规范的 OPL 结果

修改 `scripts/run_opl_baselines.py`：

- 保留当前输出：`opl_metrics.csv`, `opl_values.csv`, `opl_weights/*.csv`；
- 新增每个方法最终 wealth、平均 turnover；
- 在 metrics 中区分：`OPL`, `Hindsight Upper Bound`；
- 可选新增 `Regret_vs_BCRP`：

$$
Regret_{BCRP} = \log(V^{BCRP}_T) - \log(V^{method}_T)
$$

### A4. 验收命令

```bash
python scripts/run_opl_baselines.py --config configs/dow30.yaml
python scripts/run_opl_baselines.py --config configs/nas100.yaml
python scripts/run_opl_baselines.py --config configs/olps.yaml
python scripts/summarize_results.py
```

验收标准：

```text
results/baselines/*/opl_metrics.csv 存在
results/baselines/*/opl_values.csv 存在
results/baselines/*/opl_weights/*.csv 存在
无 NaN/inf 权重
所有权重行和约等于 1
BCRP 明确标注为 hindsight upper bound
```

---

## 5. 任务 B：补技术指标状态 X_tech

### B1. 新增 `qf_oplrl/technical_indicators.py`

实现以下函数：

```python
def moving_average_ratio(prices, window):
    # 输出 P_t / MA_window(t) - 1


def rolling_volatility(returns, window=20, annualize=False):
    # 输出 trailing std，默认不年化，用于 state


def rsi(prices, window=14):
    # 输出 RSI，并归一化到 [-1, 1] 或 [0, 1]


def macd(prices, fast=12, slow=26, signal=9):
    # 输出 macd_line, macd_signal, macd_hist，可做尺度归一化


def build_technical_feature_package(prices, config):
    # 返回 dict[str, pd.DataFrame]
```

建议输出特征：

```text
ma_ratio_5
ma_ratio_20
rsi_14
macd_line
macd_signal
macd_hist
volatility_20
```

### B2. 技术指标必须避免未来信息泄漏

原则：

```text
如果第 t 日 reward 使用 r_t，那么 observation 里不能含有第 t 日收盘后才知道的信息。
```

当前项目中 returns 是由 `prices.pct_change()` 得到。如果环境在 `current_step` 处使用 `returns.iloc[current_step]` 结算收益，则 observation 最好使用 `current_step - 1` 及以前的信息。

实现建议：

- 技术指标先基于 close prices 计算；
- 在进入 RL env 前或 env 内部统一 `shift(1)`；
- 与 QPL features 一样，使用 lagged features 与 returns index 对齐；
- 不要让 MA/RSI/MACD 使用当前待交易日的未来 close。

### B3. 修改 `plain_rl_env.py`

新增参数：

```python
use_technical_state: bool = False
technical_features: dict[str, pd.DataFrame] | None = None
technical_feature_names: list[str] | None = None
```

修改 observation：

```text
原来：lookback returns + previous_weights
升级：lookback returns + previous_weights + selected technical features
```

obs_dim 需要动态计算：

```text
obs_dim = lookback_window * n_assets + n_assets + n_tech_features * n_assets
```

注意：

- 技术指标要与 returns index、columns 对齐；
- NaN 行要统一 drop；
- 如果 `use_technical_state=True` 但 features 为空，应抛出清晰错误。

### B4. 修改 `qpl_rl_env.py`

在 QPLPortfolioEnv 中同样新增：

```python
use_technical_state: bool = False
technical_features: dict[str, pd.DataFrame] | None = None
technical_feature_names: list[str] | None = None
```

最终 state 应支持：

```text
lookback returns
+ previous weights
+ technical indicators
+ QPL d_plus, d_minus, z_qpl
```

也就是 PDF 中的：

$$
s_t = \{X^{price}_t, X^{tech}_t, X^{qpl}_t, w_{t-1}\}
$$

### B5. 修改训练入口

修改：

```text
qf_oplrl/plain_rl.py
qf_oplrl/qpl_rl.py
scripts/run_plain_rl.py
scripts/run_qpl_ablation.py
```

要求：

- 根据 config 中的 `technical_indicators.enabled` 决定是否构建技术指标；
- Plain PPO 可以跑两种版本：
  - `Plain PPO`
  - `Plain PPO + Tech State`
- QPL ablation 可以跑：
  - `PPO + Tech State + QPL State`
  - `Full QF-OPLRL V2`

### B6. 修改 configs

在 `configs/dow30.yaml`, `configs/nas100.yaml`, `configs/olps.yaml` 中新增：

```yaml
technical_indicators:
  enabled: true
  shift_features: true
  features:
    ma_ratio:
      windows: [5, 20]
    rsi:
      window: 14
    macd:
      fast: 12
      slow: 26
      signal: 9
    volatility:
      window: 20
      annualize: false
  clip_value: 10.0
```

### B7. 验收命令

```bash
python scripts/run_plain_rl.py --config configs/dow30.yaml --timesteps 5000
python scripts/run_qpl_ablation.py --config configs/dow30.yaml --first-only --timesteps 5000
```

验收标准：

```text
Plain PPO 原版仍能跑
Plain PPO + Tech State 能跑
QPL State + Tech State 能跑
observation_space shape 与实际 observation 长度一致
无 NaN/inf observation
```

---

## 6. 任务 C：升级 QPL Gate V2

### C1. 当前 Gate V1 的问题

当前 `qpl_gate.py` 的逻辑是：

```text
qpl_signal == 1  -> support_boost
qpl_signal == -1 -> resistance_cut
qpl_signal == -2 -> breakdown_cut
else             -> neutral_multiplier
```

这能体现基础 QPL 风险控制，但不够规范：

- 只有离散档位，不能表达“离 QPL 越近风险越大”；
- 主要依赖 close price，没有利用 high/low 是否盘中触碰 QPL；
- 不区分 RL 是想加仓还是减仓；
- 没有加入 volatility；
- 没有加入 portfolio drawdown；
- 没有模糊门控思想。

### C2. 新增 `qf_oplrl/qpl_gate_v2.py`

实现一个独立 V2，不要删除 V1。新增函数建议：

```python
def compute_qpl_gate_scores(
    raw_weights,
    previous_weights,
    qpl_feature_row,
    tech_feature_row=None,
    portfolio_drawdown=0.0,
    qpl_config=None,
):
    """Return per-asset continuous multipliers g_i,t."""


def apply_qpl_gate_v2_to_weight_vector(
    raw_weights,
    previous_weights,
    qpl_feature_row,
    tech_feature_row=None,
    portfolio_drawdown=0.0,
    qpl_config=None,
):
    """Apply V2 gate and re-normalize long-only weights."""
```

### C3. Gate V2 推荐公式

保留核心归一化公式：

$$
w_{i,t} = \frac{g_{i,t}\tilde{w}_{i,t}}{\sum_j g_{j,t}\tilde{w}_{j,t}}
$$

但把 multiplier 改成连续型：

$$
g_{i,t} = clip(1 + A_i - R_i - B_i - V_i - D_i, g_{min}, g_{max})
$$

其中：

```text
A_i = support_score * positive_momentum_score * add_intent_score * alpha_support
R_i = resistance_score * weak_momentum_score * add_intent_score * beta_resistance
B_i = breakdown_score * weak_momentum_score * gamma_breakdown
V_i = high_volatility_score * delta_volatility
D_i = portfolio_drawdown_score * eta_drawdown
```

关键变量：

```text
add_intent_score = max(raw_weight_i - previous_weight_i, 0)
reduce_intent_score = max(previous_weight_i - raw_weight_i, 0)
support_score = function(d_minus)
resistance_score = function(d_plus)
breakdown_score = 1 if price < qpl_minus else 0
positive_momentum_score = max(momentum_i, 0) normalized
weak_momentum_score = max(-momentum_i, 0) normalized
high_volatility_score = volatility_i / rolling_median_volatility_i
portfolio_drawdown_score = current_drawdown
```

直观逻辑：

```text
如果 RL 想加仓，并且资产接近上方 QPL + 动量变弱 -> 强烈压低权重
如果 RL 想加仓，并且资产跌破下方 QPL + 高波动 -> 强烈压低权重
如果资产接近下方 QPL + 动量转正 + 波动可控 -> 允许小幅提升权重
如果 RL 想减仓，且资产在 resistance 或 breakdown 区域 -> 不阻止减仓
如果组合当前 drawdown 已经偏大 -> 所有高风险资产更保守
```

### C4. High/Low 触碰 QPL 的可选增强

如果当前数据有 OHLC：

```text
high_t >= qpl_plus_t  -> touched_resistance = 1
low_t <= qpl_minus_t  -> touched_support = 1
```

如果数据只有 close：

```text
使用 close-based near_support / near_resistance fallback
```

Codex 可以选择轻量修改 `data_loader.py`：

- 保留当前 `prices` close matrix；
- 可选新增 `open_prices`, `high_prices`, `low_prices`, `close_prices`；
- 如果源 CSV 没有 OHLC，则这些字段为 None；
- 不能破坏现有 API。

### C5. 配置项

在 config 中新增：

```yaml
qpl_gate_v2:
  enabled: true
  mode: continuous_action_aware
  g_min: 0.30
  g_max: 1.20
  alpha_support: 0.15
  beta_resistance: 0.35
  gamma_breakdown: 0.50
  delta_volatility: 0.10
  eta_drawdown: 0.20
  support_band: 0.01
  resistance_band: 0.01
  volatility_window: 20
  use_high_low_touch: true
  fallback_to_close_touch: true
```

### C6. 修改 `qpl_rl_env.py`

新增参数：

```python
use_qpl_gate_v2: bool = False
```

step 里逻辑：

```text
if use_qpl_gate_v2:
    weights = apply_qpl_gate_v2_to_weight_vector(
        raw_weights=raw_weights,
        previous_weights=self.previous_weights,
        qpl_feature_row=current_qpl_features,
        tech_feature_row=current_tech_features,
        portfolio_drawdown=current_drawdown_before_trade,
        qpl_config=self.qpl_config,
    )
elif use_qpl_gate:
    weights = apply_qpl_gate_to_weight_vector(...)
else:
    weights = raw_weights
```

注意：

- 不要让 V2 覆盖 V1；
- ablation 要能同时比较 Gate V1 和 Gate V2；
- V2 必须保持 long-only, sum-to-one；
- multiplier 应输出到 `info`，方便 debug 和画图。

### C7. 新增 Gate V2 ablation

修改 `qf_oplrl/qpl_rl.py` 或新增脚本，使 ablation 包括：

```text
Plain PPO
Plain PPO + Tech State
PPO + QPL State
PPO + QPL Gate V1
PPO + QPL Gate V2
PPO + QPL State + Gate V2
Full QF-OPLRL V1
Full QF-OPLRL V2
```

其中：

```text
Full QF-OPLRL V2 = Tech State + QPL State + QPL Gate V2 + QPL Reward
```

### C8. Gate V2 验收命令

```bash
python scripts/run_qpl_ablation.py --config configs/dow30.yaml --first-only --timesteps 5000
python scripts/summarize_results.py
```

可选完整运行：

```bash
python scripts/run_qpl_ablation.py --config configs/dow30.yaml --timesteps 20000
python scripts/run_qpl_ablation.py --config configs/nas100.yaml --timesteps 20000
python scripts/summarize_results.py
```

验收标准：

```text
Gate V1 和 Gate V2 都能跑
Full QF-OPLRL V2 有单独结果行
输出 metrics 包含 MDD, Turnover, Sharpe, Calmar
info 或 debug 文件中能看到 gate multipliers
Gate V2 权重无 NaN/inf，且每行权重和为 1
```

---

## 7. 任务 D：结果汇总和报告支撑

### D1. 修改 `summarize_results.py`

确保最终汇总中至少包含：

```text
Dataset
Method
Method Type
Cumulative Return
Annualized Return
Annualized Volatility
Sharpe Ratio
Sortino Ratio
Maximum Drawdown
Calmar Ratio
Average Turnover
Total Transaction Cost
Win Rate
Regret_vs_BCRP   # 可选但推荐
```

### D2. 新增实验图

在 `plots.py` 或脚本中补：

```text
1. portfolio value curves
2. drawdown curves
3. final metrics bar charts
4. turnover comparison
5. Gate V1 vs Gate V2 multiplier heatmap
6. QPL trigger points 示例图
```

### D3. 最终结果目录建议

```text
results/
├── baselines/
│   └── ...
├── rl/
│   └── ...
├── qpl/
│   └── ...
├── qpl_ablation/
│   └── ...
├── summary/
│   ├── final_experiment_metrics.csv
│   ├── final_experiment_metrics.xlsx  # 可选
│   ├── portfolio_value_comparison.png
│   ├── drawdown_comparison.png
│   ├── qpl_gate_v1_v2_comparison.png
│   └── qpl_trigger_case_study.png
```

---

## 8. 推荐执行顺序

不要同时大改所有模块。按下面顺序执行，每一步都能单独跑通。

### Phase 1: OPL 标准化

```text
目标：让 BCRP/PAMR/OLMAR/ONS 更规范，输出更清楚。
参考：universal-portfolios
涉及文件：opl_baselines.py, run_opl_baselines.py, metrics.py, tests
验收：三个 config 的 OPL baseline 能跑完。
```

### Phase 2: 技术指标状态

```text
目标：补齐 X_tech，让 state 接近 PDF 愿景。
参考：FinRL
涉及文件：technical_indicators.py, plain_rl_env.py, qpl_rl_env.py, plain_rl.py, qpl_rl.py, configs
验收：Plain PPO + Tech State 和 QPL State + Tech State 能跑。
```

### Phase 3: QPL Gate V2

```text
目标：把 Gate 从离散 multiplier 升级为 continuous + action-aware + volatility-aware + drawdown-aware。
参考：Chaotic-Quantum-Finance-Trading-System, MILLION, FinRL
涉及文件：qpl_gate_v2.py, qpl_rl_env.py, qpl_rl.py, run_qpl_ablation.py, configs
验收：Gate V1 vs Gate V2 ablation 能跑。
```

### Phase 4: 结果汇总和图

```text
目标：支撑最终报告。
涉及文件：summarize_results.py, plots.py
验收：final_experiment_metrics.csv 和关键图生成。
```

---

## 9. 最终必须跑通的命令

先跑小实验：

```bash
python scripts/run_data_check.py --config configs/dow30.yaml
python scripts/run_classical_baselines.py --config configs/dow30.yaml
python scripts/run_opl_baselines.py --config configs/dow30.yaml
python scripts/run_plain_rl.py --config configs/dow30.yaml --timesteps 5000
python scripts/run_qpl_baseline.py --config configs/dow30.yaml
python scripts/run_qpl_features.py --config configs/dow30.yaml
python scripts/run_qpl_ablation.py --config configs/dow30.yaml --first-only --timesteps 5000
python scripts/summarize_results.py
```

然后跑完整实验：

```bash
python scripts/run_all_basic_experiments.py
python scripts/run_all_qpl_experiments.py
python scripts/summarize_results.py
```

如果 RL 太慢，允许先只用：

```bash
--first-only --timesteps 5000
```

生成 smoke-test 结果，再逐步扩大 timesteps。

---

## 10. 最终验收标准

Codex 完成后，项目应该满足：

```text
[OPL]
- BCRP/PAMR/OLMAR/ONS 输出合法权重
- BCRP 标注为 hindsight upper bound
- OPL 结果在 DOW30/NAS100/OLPS 上能跑

[Technical State]
- 新增 MA/RSI/MACD/Volatility
- Plain PPO + Tech State 能跑
- QPL RL 环境能同时使用 Tech State + QPL State
- 所有 state 无 NaN/inf，无未来信息泄漏

[QPL Gate V2]
- Gate V1 保留
- Gate V2 新增
- Gate V2 使用连续 QPL 距离、动量、volatility、drawdown、action intent
- Gate V2 输出 long-only sum-to-one 权重
- ablation 中能比较 Gate V1 和 Gate V2

[Experiments]
- final_experiment_metrics.csv 存在
- 至少包含 Plain PPO, PPO + QPL State, PPO + QPL Gate V1, PPO + QPL Gate V2, Full QF-OPLRL V2
- 至少输出 portfolio value 和 drawdown 图
```

---

## 11. 报告中可形成的新故事线

完成上述升级后，报告可以这样讲：

```text
我们先构建传统组合、OPL 和 Plain RL baseline，保证实验横向可比；
然后参考经典 portfolio RL 做法，补充 MA/RSI/MACD/Volatility 作为普通金融状态；
接着将 QPL 距离、QPL 区间和 QPL 突破状态加入 state，测试 QPL 是否提供额外市场结构信息；
最后设计 QPL Gate V2，把 QPL 支撑/阻力、动量、波动率、回撤和 RL 调仓意图结合起来，形成风险感知执行层。
```

最终可以对应三个研究问题：

```text
RQ1: QPL State 是否提升收益和 Sharpe？
RQ2: QPL Gate，尤其 Gate V2，是否降低 MDD 和 Turnover？
RQ3: QPL Reward + Gate V2 是否让策略更稳定？
```

---

## 12. 注意事项

1. 不要把参考项目的大量代码直接复制进当前项目。
2. 不要破坏当前能跑的 baseline 脚本。
3. 所有新增功能都要通过 config 开关控制。
4. 所有特征必须严格避免未来信息泄漏。
5. Gate V2 是新增，不是替换 Gate V1。
6. ONS 如果只实现 diagonal version，报告和结果表中必须明确写 `ONS Diagonal`。
7. BCRP 是理论上界，不要把它描述成真实可部署策略。
8. 如果某些数据集没有 OHLC，只用 close fallback，不要报错中断。

---

## 13. Codex 最终交付物

Codex 修改完成后，需要给出：

```text
1. 修改文件列表
2. 新增文件列表
3. 每个缺口对应的解决方案说明
4. 已运行的命令
5. 生成的结果路径
6. 如果有失败或跳过的实验，说明原因
```

最理想的最终结果是：

```text
qf_oplrl/technical_indicators.py
qf_oplrl/qpl_gate_v2.py
configs/*.yaml 中新增 technical_indicators 和 qpl_gate_v2 配置
scripts/run_qpl_ablation.py 能输出 Gate V1 vs Gate V2 对比
results/summary/final_experiment_metrics.csv 能支撑最终报告
```
