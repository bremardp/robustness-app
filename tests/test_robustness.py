"""Tests for the TradeStation robustness analyzer."""
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.metrics import compute_metrics, equity_curve, max_drawdown, pnl_array
from app.parser import parse_tradestation_trade_export
from app.robustness import run_robustness_suite
from app.report import build_report

DATA = Path(__file__).resolve().parent.parent / "sample_data"


def _load(name: str):
    return (DATA / name).read_bytes()


def _pnls(name: str):
    trades, err = parse_tradestation_trade_export(_load(name), name)
    assert err is None, err
    return trades, pnl_array(trades)


def test_parse_all_three_layouts_match():
    _, p1 = _pnls("sample_trades_list.csv")
    _, p2 = _pnls("sample_tradedata.csv")
    _, p3 = _pnls("sample_flat_trades.csv")
    assert len(p1) == 260
    # trades-list and TradeData CSVs store P/L rounded to cents; flat keeps full precision
    assert np.allclose(np.sort(p1), np.sort(p3), atol=1.5)
    assert np.allclose(np.sort(p2), np.sort(p3), atol=1.5)


def test_parse_errors_on_garbage():
    trades, err = parse_tradestation_trade_export(b"not a csv at all\njust some text\n", "bad.csv")
    assert trades == []
    assert err is not None


def test_parse_open_profit_layout_fallback():
    """Title row says 'TradeStation Trades List' but it is the open-profit layout."""
    content = (
        "TradeStation Trades List,,,,,,,,,\n"
        ",,,,,,,,,\n"
        "Trade Number,Type,Date,open profit,Price,,,,,\n"
        ",,,,,,,,,\n"
        "1,Sell Short,27-3-2007 ,,13.57 ,,,,,\n"
        ",,28-3-2007 ,-123.20 ,,,,,,\n"
        ",Buy to Cover,4-4-2007 ,,13.25 ,,,,,\n"
        ",,,,,,,,,\n"
        "2,Sell Short,4-4-2007 ,,13.25 ,,,,,\n"
        ",,5-4-2007 ,-134.40 ,,,,,,\n"
        ",Buy to Cover,24-4-2007 ,,12.90 ,,,,,\n"
    )
    trades, err = parse_tradestation_trade_export(content.encode(), "open.csv")
    assert err is None, err
    assert len(trades) == 2
    assert trades[0]["pnl"] == pytest.approx(-0.32)
    assert trades[1]["pnl"] == pytest.approx(-0.35)


def test_metrics_values():
    trades, pnls = _pnls("sample_flat_trades.csv")
    m = compute_metrics(trades)
    assert m["n_trades"] == 260
    assert abs(m["net_profit"] - float(pnls.sum())) < 1e-6
    assert 0 <= m["win_rate"] <= 1
    assert m["profit_factor"] > 0
    assert m["max_drawdown"] > 0
    assert m["expectancy"] == pytest.approx(float(pnls.mean()))


def test_equity_and_drawdown():
    pnls = np.array([100.0, -50.0, 200.0, -300.0, 100.0])
    eq = equity_curve(pnls)
    dd_usd, dd_pct, _, _ = max_drawdown(eq)
    # peak 250 -> trough -50 => dd of 300 (300/250 = 120%)
    assert dd_usd == pytest.approx(300.0)
    assert dd_pct == pytest.approx(1.2)


def test_robustness_suite_deterministic_and_keys():
    trades, _ = _pnls("sample_flat_trades.csv")
    r1 = run_robustness_suite(trades, n_sims=500, seed=7)
    r2 = run_robustness_suite(trades, n_sims=500, seed=7)
    assert r1["score"] == r2["score"]
    assert r1["verdict"] in ("pass", "warn", "fail")
    ids = [t["id"] for t in r1["tests"]]
    assert "monte-carlo-reshuffle" in ids
    assert "monte-carlo-bootstrap" in ids
    assert "random-trade-skip" in ids
    assert "profit-factor-stability" in ids
    assert "drawdown-consistency" in ids
    assert "bootstrap-mean-ci" in ids
    assert "risk-of-ruin" in ids
    assert "min-trades" in ids
    for t in r1["tests"]:
        assert t["status"] in ("pass", "warn", "fail")
        assert 0.0 <= t["score"] <= 1.0


def test_build_report_full_shape():
    report = build_report(_load("sample_trades_list.csv"), "sample_trades_list.csv", n_sims=300, seed=1)
    assert report["ok"] is True
    assert report["n_trades"] == 260
    assert report["summary"]["net_profit"] > 0
    assert len(report["series"]["equity"]) == 261
    assert len(report["trades"]) == 260
    assert report["robustness"]["score"] > 0


def test_build_report_error_case():
    report = build_report(b"not a csv\njust text\n", "bad.csv")
    assert report["ok"] is False


client = TestClient(app)


def test_api_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_analyze_upload():
    with open(DATA / "sample_trades_list.csv", "rb") as f:
        resp = client.post(
            "/api/analyze",
            files={"file": ("sample.csv", f, "text/csv")},
            data={"n_sims": "300", "seed": "42", "starting_equity": "100000"},
        )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ok"] is True
    assert data["n_trades"] == 260
    assert data["robustness"]["score"] > 0


def test_api_analyze_bad_file():
    resp = client.post("/api/analyze", files={"file": ("bad.csv", b"not a csv\njust text\n", "text/csv")})
    assert resp.status_code == 422
