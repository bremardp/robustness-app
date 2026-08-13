"""Generate sample TradeStation trade CSVs for the robustness app demo."""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

def make_pnls(n):
    # edge: 55% winners, avg win 900, avg loss 620
    wins = rng.uniform(300, 2200, size=n)
    losses = rng.uniform(150, 1400, size=n)
    mask = rng.random(n) < 0.55
    return np.where(mask, wins, -losses)

n = 260
pnls = make_pnls(n)

# --- TS Trades List paired-row format ---
dates = pd.date_range("2020-01-02 09:00", periods=n * 3, freq="4h")
rows = []
for i, p in enumerate(pnls):
    e = dates[i * 3]
    x = dates[i * 3 + 1]
    entry_side = rng.choice(["Buy", "Sell Short"])
    exit_side = "Sell" if entry_side == "Buy" else "Buy to Cover"
    # net profit on entry row (TS convention), gross on exit "Shares/Ctrts" cell
    rows.append(f"{i+1},{entry_side},{e.strftime('%d/%m/%Y %H:%M')},Entry,{100.0:.2f},,1,{p - 10:.2f},{p/max(abs(p),1)*50:.1f}%,,{abs(p)*0.1:.0f},,$2.50,$50.00")
    rows.append(f",{exit_side},{x.strftime('%d/%m/%Y %H:%M')},Exit,{98.0:.2f},,${p:.2f},,,{abs(p)*0.2:.0f}%,,$2.50,$50.00")
hdr = ("TradeStation Trades List\n"
       "#,Type,Date/Time,Signal,Price,Roll Over Pips,Shares/Ctrts - Profit/Loss,"
       "Net Profit - Cum Net Profit,% Profit,Run-up/Drawdown,Efficiency,Total Eff.,Comm.,Slippage\n")
text = hdr + "\n".join(rows) + "\n"
(OUT / "sample_trades_list.csv").write_text(text, encoding="utf-8")

# --- MultiWalk TradeData format ---
td_rows = []
for i, p in enumerate(pnls):
    e = dates[i * 3]
    x = dates[i * 3 + 1]
    side = rng.choice(["Long", "Short"])
    e_time = int(e.strftime("%H%M"))
    x_time = int(x.strftime("%H%M"))
    td_rows.append(f"{e.strftime('%m/%d/%Y')},{e_time},{100.0:.2f},Entry,{side},0.00,0.00,0.00")
    td_rows.append(f"{x.strftime('%m/%d/%Y')},{x_time},{98.0:.2f},Exit,{side},{p:.2f},{abs(p)*1.2:.2f},{abs(p)*0.8:.2f}")
td_hdr = "Date,Time,Price,Type,Position,PNL,MAE,MFE\n"
(OUT / "sample_tradedata.csv").write_text(td_hdr + "\n".join(td_rows) + "\n", encoding="utf-8")

# --- Flat one-row-per-trade format ---
flat = pd.DataFrame({
    "Entry Date": [d.strftime("%m/%d/%Y") for d in dates[::3]],
    "Entry Time": [int(d.strftime("%H%M")) for d in dates[::3]],
    "Exit Date": [d.strftime("%m/%d/%Y") for d in dates[1::3]],
    "Exit Time": [int(d.strftime("%H%M")) for d in dates[1::3]],
    "Net Profit": pnls,
})
flat.to_csv(OUT / "sample_flat_trades.csv", index=False)

print("net pnl:", round(float(pnls.sum())), "trades:", n)
for f in sorted(OUT.glob("*.csv")):
    print(f.name, f.stat().st_size, "bytes")
