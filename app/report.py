"""
Assemble the full analysis report: parse -> metrics -> robustness suite.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from app import metrics as metrics_mod
from app import robustness
from app.parser import parse_tradestation_trade_export


def sanitize_for_json(obj: Any) -> Any:
    """Replace non-finite floats (inf/nan) with None so the report is JSON-safe."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def _jsonable_trades(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for t in trades:
        row: Dict[str, Any] = {"pnl": round(float(t["pnl"]), 4)}
        for k in ("entry", "exit"):
            v = t.get(k)
            row[k] = None if v is None else str(v)
        for k in (
            "entry_price",
            "exit_price",
            "entry_type",
            "exit_type",
            "pnl_gross",
            "pnl_net",
            "ts_commission",
            "ts_slippage",
        ):
            v = t.get(k)
            row[k] = None if v is None else v
        out.append(row)
    return out


def build_report(
    content: bytes,
    filename: str,
    *,
    n_sims: int = 2000,
    seed: Optional[int] = None,
    starting_equity: float = 100_000.0,
    ruin_point: Optional[float] = None,
    skip_fraction: float = 0.10,
) -> Dict[str, Any]:
    trades, err = parse_tradestation_trade_export(content, filename)
    if err:
        return {"ok": False, "error": err}
    if not trades:
        return {"ok": False, "error": "No trades could be parsed from the file."}

    pnls = metrics_mod.pnl_array(trades)
    summary = metrics_mod.compute_metrics(trades)
    suite = robustness.run_robustness_suite(
        trades,
        n_sims=n_sims,
        seed=seed,
        starting_equity=starting_equity,
        ruin_point=ruin_point,
        skip_fraction=skip_fraction,
    )

    eq = metrics_mod.equity_curve(pnls)
    dd_usd, dd_pct, _, _ = metrics_mod.max_drawdown(eq)
    eq_dollars = eq + (starting_equity if starting_equity else 0.0)
    drawdown_curve = np.maximum.accumulate(eq) - eq

    summary["starting_equity"] = starting_equity

    return sanitize_for_json({
        "ok": True,
        "filename": filename,
        "n_trades": len(trades),
        "parsed_as": _parsed_layout_hint(trades),
        "summary": summary,
        "robustness": suite,
        "series": {
            "equity": [round(float(x), 4) for x in eq_dollars],
            "cumulative_pnl": [round(float(x), 4) for x in eq],
            "drawdown": [round(float(x), 4) for x in drawdown_curve],
            "trade_pnl": [round(float(x), 4) for x in pnls],
        },
        "trades": _jsonable_trades(trades),
        "settings": {
            "n_sims": suite["n_sims"],
            "seed": suite["seed"],
            "starting_equity": starting_equity,
            "skip_fraction": skip_fraction,
        },
    })

def _parsed_layout_hint(trades: List[Dict[str, Any]]) -> str:
    if trades and "pnl_gross" in trades[0]:
        return "TradeStation Trades List (paired entry/exit rows)"
    if trades and "ts_commission" in trades[0]:
        return "TradeStation Trades List (paired entry/exit rows)"
    if trades and "daily_profits" in trades[0]:
        return "TradeStation Trades List (open profit)"
    if trades and "entry_type" in trades[0] and "signal" not in trades[0]:
        return "MultiWalk TradeData (Entry/Exit events)"
    if trades and any("entry_price" in t for t in trades):
        return "Flat trade table with entry/exit timestamps"
    return "Flat trade table with entry/exit timestamps"
