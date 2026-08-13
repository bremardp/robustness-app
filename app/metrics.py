"""
Summary performance metrics computed from a closed-trade list.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def pnl_array(trades: List[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(t["pnl"]) for t in trades], dtype=float)


def equity_curve(pnls: np.ndarray, start: float = 0.0) -> np.ndarray:
    """Cumulative P/L curve; index i = equity after trade i (i starts at 1)."""
    return np.concatenate([[start], start + np.cumsum(pnls)])


def max_drawdown(equity: np.ndarray) -> Tuple[float, float, int, int]:
    """(max_drawdown_$, max_drawdown_pct_of_peak, peak_idx, trough_idx)."""
    if equity.size == 0:
        return 0.0, 0.0, 0, 0
    running_max = np.maximum.accumulate(equity)
    dd_dollar = running_max - equity
    max_idx = int(np.argmax(dd_dollar))
    peak_idx = int(np.argmax(equity[: max_idx + 1])) if max_idx > 0 else max_idx
    peak = equity[peak_idx]
    pct = dd_dollar[max_idx] / peak if peak > 0 else 0.0
    return float(dd_dollar[max_idx]), float(pct), peak_idx, max_idx


def max_drawdown_duration(equity: np.ndarray) -> int:
    """Longest peak-to-recovery span (in trades) for the deepest dollar drawdown."""
    if equity.size < 2:
        return 0
    running_max = np.maximum.accumulate(equity)
    dd_dollar = running_max - equity
    max_idx = int(np.argmax(dd_dollar))
    peak_val = running_max[max_idx]
    peak_idx = int(np.argmax(equity[: max_idx + 1]))
    rec_idx = max_idx
    while rec_idx + 1 < equity.size and equity[rec_idx + 1] < peak_val:
        rec_idx += 1
    return max(0, rec_idx - peak_idx)


def _profit_factor(gross_profit: float, gross_loss: float) -> Optional[float]:
    if abs(gross_loss) <= 1e-12:
        return float("inf") if gross_profit > 0 else None
    return float(gross_profit / abs(gross_loss))


def compute_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics from a list of closed trades."""
    if not trades:
        return {"error": "No trades to analyze.", "n_trades": 0}

    pnls = pnl_array(trades)
    n = int(len(pnls))
    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]
    flat = pnls[pnls == 0]
    n_win = int(winners.size)
    n_loss = int(losers.size)

    net_profit = float(pnls.sum())
    gross_profit = float(winners.sum()) if n_win else 0.0
    gross_loss = float(losers.sum()) if n_loss else 0.0
    pf = _profit_factor(gross_profit, gross_loss)
    pf_infinite = pf == float("inf")

    avg_win = float(winners.mean()) if n_win else 0.0
    avg_loss = float(losers.mean()) if n_loss else 0.0
    payoff = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-12 else None
    win_rate = n_win / n if n else 0.0
    expectancy = float(pnls.mean())

    eq = equity_curve(pnls)
    dd_usd, dd_pct, peak_idx, trough_idx = max_drawdown(eq)
    dd_dur = max_drawdown_duration(eq)
    rf = net_profit / dd_usd if dd_usd > 1e-12 else (None if net_profit <= 0 else float("inf"))
    recovery_factor = rf
    recovery_infinite = rf == float("inf")

    # Streaks
    longest_win = longest_loss = 0
    cur_win = cur_loss = 0
    for p in pnls:
        if p > 0:
            cur_win += 1
            cur_loss = 0
        elif p < 0:
            cur_loss += 1
            cur_win = 0
        else:
            cur_win = cur_loss = 0
        longest_win = max(longest_win, cur_win)
        longest_loss = max(longest_loss, cur_loss)

    # Holding time (only when timestamps available)
    durations = []
    for t in trades:
        try:
            d = (t["exit"] - t["entry"]).total_seconds() / 3600.0
            if d >= 0:
                durations.append(d)
        except Exception:
            continue

    return {
        "n_trades": n,
        "net_profit": net_profit,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": pf,
        "profit_factor_infinite": pf_infinite,
        "win_rate": win_rate,
        "n_winners": n_win,
        "n_losers": n_loss,
        "n_flat": int(flat.size),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "expectancy": expectancy,
        "avg_win_expectancy": win_rate * avg_win - (1.0 - win_rate) * abs(avg_loss),
        "max_drawdown": dd_usd,
        "max_drawdown_pct": dd_pct,
        "max_drawdown_duration_trades": dd_dur,
        "recovery_factor": recovery_factor,
        "recovery_factor_infinite": recovery_infinite,
        "longest_win_streak": longest_win,
        "longest_loss_streak": longest_loss,
        "avg_bars_held_hours": float(np.mean(durations)) if durations else None,
        "median_trade": float(np.median(pnls)),
        "stddev_trade": float(np.std(pnls, ddof=1)) if n > 1 else 0.0,
    }
