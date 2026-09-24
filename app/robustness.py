"""
Trade-list robustness tests.

All tests operate on the closed-trade P/L list only (no OHLC data needed):

1. min-trades                - is the sample large enough for meaningful statistics
2. monte-carlo-reshuffle     - randomize trade order, recompute equity path
3. monte-carlo-bootstrap     - resample trades with replacement (ruin risk, median DD)
4. random-trade-skip         - drop a random fraction of trades, sensitivity of net P/L
5. profit-factor-stability   - bootstrap distribution of profit factor
6. drawdown-consistency      - is the realized max drawdown unusual vs random order
7. bootstrap-mean-ci         - 95% CI on mean trade P/L (expectancy)
8. risk-of-ruin              - probability equity falls below a floor (from bootstrap)
9. r-squared-slope-equity    - how closely the equity curve follows a straight-line trend

Every test returns a dict::

    {id, name, description, status, score, details, metrics}
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.metrics import equity_curve, max_drawdown, pnl_array

DEFAULT_SIMS = 2000
MAX_MATRIX_CELLS = int(2.5e7)  # cap n_sims * n_trades for the vectorized matrix


def _adapt_sims(n_sims: int, n_trades: int) -> int:
    if n_trades <= 0:
        return n_sims
    cap = max(200, MAX_MATRIX_CELLS // n_trades)
    return max(50, min(int(n_sims), cap))


def _sample_equity_paths(
    pnls: np.ndarray,
    n_sims: int,
    seed: int,
    replace: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cum_paths, final_equities, max_dollar_drawdowns) for sampled trade orders."""
    n = int(pnls.size)
    rng = np.random.default_rng(seed)
    n_sims = _adapt_sims(n_sims, n)
    if n == 0:
        return np.zeros((n_sims, 1)), np.zeros(n_sims), np.zeros(n_sims)
    if replace:
        idx = rng.integers(0, n, size=(n_sims, n))
    else:
        idx = np.argsort(rng.random((n_sims, n)), axis=1)
    sampled = pnls[idx]
    cum = np.cumsum(sampled, axis=1)
    # Include the starting 0 point so drawdown is measured from the true peak
    # (consistent with metrics.max_drawdown on the realized equity curve).
    cum0 = np.concatenate([np.zeros((n_sims, 1)), cum], axis=1)
    running_max = np.maximum.accumulate(cum0, axis=1)
    dd = running_max - cum0
    max_dd = dd.max(axis=1)
    final = cum[:, -1]
    return cum, final, max_dd


def _thin(series: np.ndarray, points: int = 200) -> List[float]:
    if series.ndim == 2:
        series = series.mean(axis=0)
    s = np.asarray(series, dtype=float)
    if s.size <= points:
        return [float(x) for x in s]
    idx = np.linspace(0, s.size - 1, points).astype(int)
    return [float(x) for x in s[idx]]


def _percentiles(data: np.ndarray, qs) -> Dict[str, float]:
    if data.size == 0:
        return {f"p{int(q * 100)}": 0.0 for q in qs}
    vals = np.percentile(data, [q * 100 for q in qs])
    return {f"p{int(q * 100)}": float(v) for q, v in zip(qs, vals)}


def _result(
    tid: str,
    name: str,
    description: str,
    status: str,
    score: float,
    details: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "id": tid,
        "name": name,
        "description": description,
        "status": status,
        "score": round(float(score), 3),
        "details": details,
        "metrics": metrics or {},
    }


# ---------------------------------------------------------------------------
# 1. Minimum trades
# ---------------------------------------------------------------------------
def min_trades_test(pnls: np.ndarray, n_sims: int, seed: int) -> Dict[str, Any]:
    n = int(pnls.size)
    if n < 30:
        status, score = "fail", 0.0
        detail = (
            f"Only {n} trades. Below the 30-trade minimum, statistical conclusions are unreliable."
        )
    elif n < 100:
        status, score = "warn", 0.5
        detail = f"{n} trades: adequate but a larger sample would tighten confidence intervals (target >= 100)."
    else:
        status, score = "pass", 1.0
        detail = f"{n} trades: a solid sample for trade-list statistics."
    mean = float(np.mean(pnls)) if n else 0.0
    std = float(np.std(pnls, ddof=1)) if n > 1 else 0.0
    ci_half = 1.96 * std / (n ** 0.5) if n > 1 else float("inf")
    need_20pct = (
        int(((1.96 * std) / (0.20 * abs(mean))) ** 2) + 1
        if abs(mean) > 1e-12 and std > 0
        else None
    )
    return _result(
        "min-trades",
        "Minimum Trades",
        "Statistical adequacy of the trade sample.",
        status,
        score,
        detail,
        {
            "n_trades": n,
            "mean_trade": mean,
            "ci_half_width_95": ci_half if np.isfinite(ci_half) else None,
            "trades_needed_for_20pct_ci": need_20pct,
        },
    )


# ---------------------------------------------------------------------------
# 2. Monte Carlo reshuffle (order randomization)
# ---------------------------------------------------------------------------
def monte_carlo_reshuffle_test(pnls: np.ndarray, n_sims: int, seed: int) -> Dict[str, Any]:
    n = int(pnls.size)
    actual_final = float(np.sum(pnls))
    cum, final, max_dd = _sample_equity_paths(pnls, n_sims, seed, replace=False)
    prob_positive = float(np.mean(final > 0))
    actual_dd = max_drawdown(equity_curve(pnls))[0]
    pct_sims_dd_gt_actual = float(np.mean(max_dd > actual_dd)) if actual_dd > 1e-12 else 1.0

    if prob_positive >= 0.95:
        status, score = "pass", 1.0
    elif prob_positive >= 0.75:
        status, score = "warn", 0.5
    else:
        status, score = "fail", 0.0
    detail = (
        f"{prob_positive * 100:.1f}% of {n_sims} random orderings end profitable "
        f"(actual net P/L ${actual_final:,.2f}). "
        f"{pct_sims_dd_gt_actual * 100:.1f}% of random orderings hit a deeper max drawdown "
        f"than the realized ${actual_dd:,.2f}."
    )
    return _result(
        "monte-carlo-reshuffle",
        "Monte Carlo Reshuffle (Order Randomization)",
        "Randomizes trade order to test whether results depend on a lucky sequence.",
        status,
        score,
        detail,
        {
            "n_sims": _adapt_sims(n_sims, n),
            "actual_net_profit": actual_final,
            "prob_positive": prob_positive,
            "final_equity_percentiles": _percentiles(final, [0.05, 0.25, 0.5, 0.75, 0.95]),
            "max_dd_percentiles": _percentiles(max_dd, [0.05, 0.5, 0.95]),
            "pct_sims_dd_gt_actual": pct_sims_dd_gt_actual,
            "equity_percentile_path": {
                "p5": _thin(np.percentile(cum, 5, axis=0)),
                "p50": _thin(np.percentile(cum, 50, axis=0)),
                "p95": _thin(np.percentile(cum, 95, axis=0)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 3. Monte Carlo bootstrap (with replacement) + ruin
# ---------------------------------------------------------------------------
def monte_carlo_bootstrap_test(
    pnls: np.ndarray,
    n_sims: int,
    seed: int,
    starting_equity: float,
    ruin_point: float,
) -> Dict[str, Any]:
    n = int(pnls.size)
    start = float(starting_equity or 100_000.0)
    ruin = float(ruin_point if ruin_point and ruin_point > 0 else 0.6 * start)

    cum, final, max_dd = _sample_equity_paths(pnls, n_sims, seed, replace=True)
    equity = start + cum
    ruin_path = (equity <= ruin).any(axis=1)
    ruin_prob = float(np.mean(ruin_path))
    final_eq = start + final
    returns_pct = (final / start) * 100 if start > 0 else np.zeros_like(final)
    running_max_eq = np.maximum.accumulate(equity, axis=1)
    dd_pct_sims = ((running_max_eq - equity) / np.where(running_max_eq > 0, running_max_eq, 1.0)).max(axis=1)
    median_dd_pct = float(np.median(dd_pct_sims))
    median_profit = float(np.median(final))
    median_return = float(np.median(returns_pct))
    prob_positive = float(np.mean(final > 0))
    return_dd_ratio = abs(median_return / (median_dd_pct * 100)) if median_dd_pct > 1e-12 else 0.0

    if ruin_prob <= 0.05:
        status, score = "pass", 1.0
    elif ruin_prob <= 0.20:
        status, score = "warn", 0.5
    else:
        status, score = "fail", 0.0
    detail = (
        f"Resampling {n} trades with replacement {_adapt_sims(n_sims, n)}x from ${start:,.0f} "
        f"starting capital (ruin floor ${ruin:,.0f}): ruin probability {ruin_prob * 100:.1f}%, "
        f"median ending profit ${median_profit:,.2f}, median max drawdown {median_dd_pct * 100:.1f}%."
    )
    return _result(
        "monte-carlo-bootstrap",
        "Monte Carlo Bootstrap (With Replacement)",
        "Resamples trades to estimate ruin probability, median drawdown and return.",
        status,
        score,
        detail,
        {
            "n_sims": _adapt_sims(n_sims, n),
            "starting_equity": start,
            "ruin_point": ruin,
            "ruin_probability": ruin_prob,
            "median_final_equity": float(np.median(final_eq)),
            "median_profit": median_profit,
            "median_return": median_return,
            "prob_positive": prob_positive,
            "median_max_dd_pct": median_dd_pct,
            "return_dd_ratio": return_dd_ratio,
            "final_equity_percentiles": _percentiles(final_eq, [0.05, 0.25, 0.5, 0.75, 0.95]),
        },
    )


# ---------------------------------------------------------------------------
# 4. Random trade skip
# ---------------------------------------------------------------------------
def random_trade_skip_test(
    pnls: np.ndarray, n_sims: int, seed: int, skip_fraction: float = 0.10
) -> Dict[str, Any]:
    n = int(pnls.size)
    frac = max(0.01, min(0.5, float(skip_fraction or 0.10)))
    keep = int(max(1, n * (1.0 - frac)))
    rng = np.random.default_rng(seed)
    n_sims = _adapt_sims(n_sims, n)
    idx = np.argsort(rng.random((n_sims, n)), axis=1)[:, :keep]
    sampled = pnls[idx]
    net = sampled.sum(axis=1)
    prob_positive = float(np.mean(net > 0))
    median_net = float(np.median(net))
    actual = float(np.sum(pnls))
    pct_change_median = (median_net / actual - 1.0) * 100 if abs(actual) > 1e-12 else 0.0

    if prob_positive >= 0.90:
        status, score = "pass", 1.0
    elif prob_positive >= 0.70:
        status, score = "warn", 0.5
    else:
        status, score = "fail", 0.0
    detail = (
        f"After randomly dropping {frac * 100:.0f}% of trades ({n_sims} runs), "
        f"{prob_positive * 100:.1f}% of outcomes remain profitable. "
        f"Median net P/L ${median_net:,.2f} vs actual ${actual:,.2f} "
        f"({pct_change_median:+.1f}% shift)."
    )
    return _result(
        "random-trade-skip",
        "Randomized Trade Skip",
        "Drops random trades to test reliance on any single trade.",
        status,
        score,
        detail,
        {
            "n_sims": n_sims,
            "skip_fraction": frac,
            "actual_net_profit": actual,
            "median_net_profit": median_net,
            "prob_positive": prob_positive,
            "net_profit_percentiles": _percentiles(net, [0.05, 0.5, 0.95]),
        },
    )


# ---------------------------------------------------------------------------
# 5. Profit factor stability
# ---------------------------------------------------------------------------
def profit_factor_stability_test(pnls: np.ndarray, n_sims: int, seed: int) -> Dict[str, Any]:
    n = int(pnls.size)
    rng = np.random.default_rng(seed)
    n_sims = _adapt_sims(n_sims, n)
    pfs: List[float] = []
    for _ in range(n_sims):
        s = pnls[rng.integers(0, n, size=n)]
        gp = s[s > 0].sum()
        gl = s[s < 0].sum()
        if abs(gl) <= 1e-12:
            pfs.append(float("inf"))
        else:
            pfs.append(float(gp / abs(gl)))
    arr = np.asarray([p for p in pfs if np.isfinite(p)], dtype=float)
    inf_count = int(sum(1 for p in pfs if np.isfinite(p) and p > 1000) + sum(1 for p in pfs if np.isinf(p)))
    if arr.size:
        median_pf = float(np.median(arr))
        p10_pf = float(np.percentile(arr, 10))
        frac_pf_above_1 = float((arr > 1.0).sum() / arr.size)
    else:
        median_pf = float("inf")
        p10_pf = float("inf")
        frac_pf_above_1 = 1.0

    if frac_pf_above_1 >= 0.95:
        status, score = "pass", 1.0
    elif frac_pf_above_1 >= 0.80:
        status, score = "warn", 0.5
    else:
        status, score = "fail", 0.0
    detail = (
        f"Across {n_sims} bootstrap samples the median profit factor is "
        f"{'inf' if not np.isfinite(median_pf) else f'{median_pf:.2f}'} "
        f"(10th percentile {'inf' if not np.isfinite(p10_pf) else f'{p10_pf:.2f}'}); "
        f"{frac_pf_above_1 * 100:.1f}% of samples keep PF > 1. "
        f"{inf_count} samples had no losing trades."
    )
    return _result(
        "profit-factor-stability",
        "Profit Factor Stability",
        "Bootstrap distribution of profit factor - does edge survive re-sampling?",
        status,
        score,
        detail,
        {
            "n_sims": n_sims,
            "median_profit_factor": median_pf if np.isfinite(median_pf) else None,
            "p10_profit_factor": p10_pf if np.isfinite(p10_pf) else None,
            "frac_runs_pf_gt_1": frac_pf_above_1,
            "samples_no_losers": inf_count,
        },
    )


# ---------------------------------------------------------------------------
# 6. Drawdown consistency
# ---------------------------------------------------------------------------
def drawdown_consistency_test(pnls: np.ndarray, n_sims: int, seed: int) -> Dict[str, Any]:
    n = int(pnls.size)
    eq = equity_curve(pnls)
    actual_dd_usd, actual_dd_pct, _, _ = max_drawdown(eq)
    _, _, max_dd_sims = _sample_equity_paths(pnls, n_sims, seed, replace=False)
    median_sim_dd = float(np.median(max_dd_sims))
    if actual_dd_usd > 1e-12:
        pctile = float(np.mean(max_dd_sims <= actual_dd_usd))
    else:
        pctile = 1.0

    if pctile <= 0.5:
        status, score = "pass", 1.0
    elif pctile <= 0.75:
        status, score = "warn", 0.5
    else:
        status, score = "fail", 0.0
    detail = (
        f"Realized max drawdown ${actual_dd_usd:,.2f} ({actual_dd_pct * 100:.1f}%) sits at the "
        f"{pctile * 100:.0f}th percentile of {n_sims} random orderings "
        f"(median simulated ${median_sim_dd:,.2f}). "
        f"{'Normal for this trade set.' if status == 'pass' else 'Drawdown is concentrated - caution.'}"
    )
    return _result(
        "drawdown-consistency",
        "Drawdown Consistency",
        "Checks whether the realized max drawdown is unusual vs random trade order.",
        status,
        score,
        detail,
        {
            "n_sims": _adapt_sims(n_sims, n),
            "actual_max_dd_usd": actual_dd_usd,
            "actual_max_dd_pct": actual_dd_pct,
            "median_simulated_max_dd_usd": median_sim_dd,
            "actual_dd_percentile": pctile,
            "simulated_max_dd_percentiles": _percentiles(max_dd_sims, [0.05, 0.5, 0.95]),
        },
    )


# ---------------------------------------------------------------------------
# 7. Bootstrap mean CI
# ---------------------------------------------------------------------------
def bootstrap_mean_ci_test(pnls: np.ndarray, n_sims: int, seed: int) -> Dict[str, Any]:
    n = int(pnls.size)
    rng = np.random.default_rng(seed)
    n_sims = _adapt_sims(n_sims, n)
    means = np.asarray([pnls[rng.integers(0, n, size=n)].mean() for _ in range(n_sims)])
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    mean = float(np.mean(pnls))

    if lo > 0:
        status, score = "pass", 1.0
        verdict = "positive at the 95% confidence level"
    elif hi < 0:
        status, score = "fail", 0.0
        verdict = "negative at the 95% confidence level - edge looks negative"
    else:
        status, score = "warn", 0.5
        verdict = "the 95% CI crosses zero - expectancy is not statistically reliable"
    detail = (
        f"Mean trade ${mean:,.2f}; 95% bootstrap CI [{lo:,.2f}, {hi:,.2f}] -> {verdict}."
    )
    return _result(
        "bootstrap-mean-ci",
        "Bootstrap Confidence Interval (Expectancy)",
        "95% CI for mean trade P/L from resampling.",
        status,
        score,
        detail,
        {"n_sims": n_sims, "mean_trade": mean, "ci_low": lo, "ci_high": hi},
    )


# ---------------------------------------------------------------------------
# 8. Risk of ruin
# ---------------------------------------------------------------------------
def risk_of_ruin_test(
    pnls: np.ndarray,
    n_sims: int,
    seed: int,
    starting_equity: float,
    ruin_point: float,
) -> Dict[str, Any]:
    n = int(pnls.size)
    start = float(starting_equity or 100_000.0)
    ruin = float(ruin_point if ruin_point and ruin_point > 0 else 0.6 * start)
    n_sims = _adapt_sims(n_sims, n)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sims):
        eq = start + np.cumsum(pnls[rng.integers(0, n, size=n)])
        if (eq <= ruin).any():
            hits += 1
    ruin_prob = hits / n_sims
    if ruin_prob <= 0.05:
        status, score = "pass", 1.0
    elif ruin_prob <= 0.20:
        status, score = "warn", 0.5
    else:
        status, score = "fail", 0.0
    detail = (
        f"From ${start:,.0f} capital, probability of touching the ${ruin:,.0f} ruin floor "
        f"within a {n}-trade run is {ruin_prob * 100:.1f}%."
    )
    return _result(
        "risk-of-ruin",
        "Risk of Ruin",
        "Probability that equity falls to a ruin floor during a simulated trade run.",
        status,
        score,
        detail,
        {"n_sims": n_sims, "starting_equity": start, "ruin_point": ruin, "ruin_probability": ruin_prob},
    )


# ---------------------------------------------------------------------------
# 9. R-Squared Slope Equity Curve
# ---------------------------------------------------------------------------
DEFAULT_MIN_R2 = 0.70
_WARN_R2 = 0.60  # warn band: 0.60 <= R2 < 0.70


def _js_round(x: float) -> float:
    """Match JavaScript ``Math.round`` semantics (round half toward +inf)."""
    f = math.floor(x)
    return f + 1.0 if (x - f) >= 0.5 else f


def _ols_fit(xs: List[float], ys: List[float]) -> Optional[Tuple[float, float]]:
    """Ordinary least-squares linear fit -> (intercept, slope) or None."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = 0.0
    den = 0.0
    for i in range(n):
        dx = xs[i] - mean_x
        num += dx * (ys[i] - mean_y)
        den += dx * dx
    if abs(den) <= 1e-12:
        return None
    slope = num / den
    return mean_y - slope * mean_x, slope


def _coefficient_of_determination(xs: List[float], ys: List[float]) -> Optional[float]:
    """R-squared of ys vs the OLS line on x -> 1 - SSres / SStot (1.0 if SStot==0)."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    fit = _ols_fit(xs, ys)
    if fit is None:
        return None
    intercept, slope = fit
    mean_y = sum(ys) / n
    ss_tot = float(sum((y - mean_y) ** 2 for y in ys))
    if ss_tot <= 1e-12:
        return 1.0
    ss_res = float(sum((ys[i] - (intercept + slope * xs[i])) ** 2 for i in range(n)))
    return 1.0 - ss_res / ss_tot


def r_squared_slope_equity_test(trades: List[Dict[str, Any]], min_r2: float = DEFAULT_MIN_R2) -> Dict[str, Any]:
    """
    How closely the equity curve follows a straight-line trend.

    Equity = cumulative net P/L in chronological (exit-time) trade order, values
    rounded like the reference tool's JavaScript ``Math.round``. An OLS line is
    fit vs trade index 1..n and R-squared measures the fraction of equity
    variance explained by that line. Pass: R2 > 0.70; warn: 0.60-0.70; else fail.
    """
    ordered = sorted(trades, key=lambda t: (t.get("exit"), t.get("entry")))
    equity: List[float] = []
    running = 0.0
    for t in ordered:
        running += float(t["pnl"])
        equity.append(_js_round(running))

    n = len(equity)
    r2: Optional[float] = None
    slope: Optional[float] = None
    if n >= 2:
        xs = [float(i) for i in range(1, n + 1)]
        r2 = _coefficient_of_determination(xs, equity)
        fit = _ols_fit(xs, equity)
        if fit is not None:
            slope = fit[1]

    if r2 is None:
        status, score = "fail", 0.0
        detail = (
            f"R-squared = n/a. At least 2 trades are required to fit a regression line "
            f"(had {n})."
        )
    elif r2 > min_r2:
        status, score = "pass", 1.0
    elif r2 >= _WARN_R2:
        status, score = "warn", 0.5
    else:
        status, score = "fail", 0.0

    if r2 is not None and slope is not None:
        verdict = (
            "the equity curve tracks a straight-line trend well." if status == "pass"
            else "the equity curve only loosely follows a straight line - growth is uneven."
        )
        detail = (
            f"Equity-curve R-squared is {r2:.4f} (pass requires > {min_r2:.2f}); "
            f"the line rises ${slope:,.2f} per trade on average, so {verdict}"
        )
    return _result(
        "r-squared-slope-equity",
        "R-Squared Slope Equity Curve",
        "R-squared of the equity curve vs trade index - how closely growth follows a straight line.",
        status,
        score,
        detail,
        {
            "r_squared": r2 if r2 is not None else None,
            "min_r2_to_pass": min_r2,
            "slope_per_trade": slope if slope is not None else None,
            "equity_points": n,
        },
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
_TEST_WEIGHTS = {
    "min-trades": 0.10,
    "monte-carlo-reshuffle": 0.15,
    "monte-carlo-bootstrap": 0.20,
    "random-trade-skip": 0.10,
    "profit-factor-stability": 0.15,
    "drawdown-consistency": 0.10,
    "bootstrap-mean-ci": 0.10,
    "risk-of-ruin": 0.10,
    "r-squared-slope-equity": 0.10,
}


def run_robustness_suite(
    trades: List[Dict[str, Any]],
    *,
    n_sims: int = DEFAULT_SIMS,
    seed: Optional[int] = None,
    starting_equity: float = 100_000.0,
    ruin_point: Optional[float] = None,
    skip_fraction: float = 0.10,
) -> Dict[str, Any]:
    pnls = pnl_array(trades)
    if pnls.size == 0:
        return {"tests": [], "score": 0.0, "verdict": "fail", "message": "No trades."}

    rng_seed = seed if seed is not None else int(np.random.default_rng().integers(0, 2**31 - 1))

    tests: List[Dict[str, Any]] = [
        min_trades_test(pnls, n_sims, rng_seed),
        monte_carlo_reshuffle_test(pnls, n_sims, rng_seed + 1),
        monte_carlo_bootstrap_test(pnls, n_sims, rng_seed + 2, starting_equity, ruin_point),
        random_trade_skip_test(pnls, n_sims, rng_seed + 3, skip_fraction),
        profit_factor_stability_test(pnls, n_sims, rng_seed + 4),
        drawdown_consistency_test(pnls, n_sims, rng_seed + 5),
        bootstrap_mean_ci_test(pnls, n_sims, rng_seed + 6),
        risk_of_ruin_test(pnls, n_sims, rng_seed + 7, starting_equity, ruin_point),
        r_squared_slope_equity_test(trades),
    ]

    weight_sum = 0.0
    score_sum = 0.0
    for t in tests:
        w = _TEST_WEIGHTS.get(t["id"], 0.1)
        weight_sum += w
        score_sum += w * t["score"]
    score = score_sum / weight_sum if weight_sum else 0.0

    worst = min(tests, key=lambda t: t["score"])
    if worst["score"] == 1.0:
        verdict = "pass"
    elif worst["score"] >= 0.5:
        verdict = "warn"
    else:
        verdict = "fail"

    return {
        "tests": tests,
        "score": round(float(score), 3),
        "verdict": verdict,
        "message": (
            f"Overall robustness {score * 100:.0f}/100 ({verdict.upper()}). "
            f"Weakest area: {worst['name']} ({worst['status'].upper()})."
        ),
        "n_sims": n_sims,
        "seed": rng_seed,
    }
