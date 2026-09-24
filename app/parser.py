"""
Parse TradeStation trade list exports (CSV) into a uniform trade list.

Supports three TradeStation layouts, auto-detected:

1. **Strategy Performance Report -> Trades List** (paired rows sharing one
   ``Date/Time`` column; header like ``#,Type,Date/Time,Signal,Price,...`` with a
   ``Net Profit`` column). Entry leg and exit leg are two consecutive rows.
2. **MultiWalk walk-forward ``TradeData.csv``**: one row per Entry/Exit event,
   columns ``Date,Time,Price,Type,Position,PNL,MAE,MFE``.
3. **Flat one-row-per-trade table** with separate Entry / Exit date-time columns
   and a numeric profit column (e.g. Net Profit, P/L).
4. **TradeStation Trades List with open profit** (``Trade Number, Type, Date,
   open profit, Price`` with daily open-profit rows between entry and exit).

Each parsed trade is a dict: ``{"entry", "exit", "pnl", ...}`` where ``entry`` and
``exit`` are ``pandas.Timestamp`` and ``pnl`` is the trade net P/L (float, dollars).

Adapted from the TradeStation trade-export parser used by the LEGO Block platform
(``tradestation_trade_verify.py``); this copy is fully standalone with no platform imports.
"""

from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _norm_header(c: Any) -> str:
    return re.sub(r"\s+", " ", str(c).strip().lower())


# TS % Profit can use thousands separators (e.g. 23,500.00%) which break naive CSV splits.
_TS_TRADES_LIST_PCT_COMMA = re.compile(r"(\d),(\d{3}(?:,\d{3})*)(\.\d+%)")


def _sanitize_ts_trades_list_csv_line(line: str) -> str:
    """Remove thousands commas inside % Profit cells so pandas reads the right column count."""
    return _TS_TRADES_LIST_PCT_COMMA.sub(r"\1\2\3", line)


def _try_decode_csv_text(content: bytes) -> Optional[str]:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return content.decode(enc)
        except Exception:
            continue
    return None


def _read_export_table(content: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "export.csv").lower()
    bio = io.BytesIO(content)

    if name.endswith(".xlsx"):
        return pd.read_excel(bio, engine="openpyxl")

    last_err: Optional[Exception] = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding=enc)
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise ValueError(f"Could not read CSV: {last_err}") from last_err
    raise ValueError("Could not read CSV.")


def _parse_currency_us(val: Any) -> float:
    """Parse TradeStation dollar cells: $1,234.56 or ($1,234.56)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return 0.0
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace("$", "").replace(",", "").strip()
    if not s:
        return 0.0
    try:
        x = float(s)
        return -abs(x) if neg else x
    except ValueError:
        return 0.0


def _parse_ts_price(val: Any) -> Optional[float]:
    """Parse TradeStation Price column (may be negative continuous-contract prices)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().replace("$", "").replace("(", "").replace(")", "")
    if not s or s.lower() in ("nan", "none"):
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _looks_like_ts_trade_profit_cell(val: Any) -> bool:
    """True when Shares/Ctrts Profit/Loss is dollar P&L (exit leg), not a contract count."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    s = str(val).strip()
    if not s:
        return False
    if s.startswith("$") or (s.startswith("(") and s.endswith(")")):
        return True
    try:
        x = float(s.replace(",", ""))
    except ValueError:
        return False
    if abs(x - round(x)) > 1e-9:
        return True
    return abs(x) >= 2.0


# ---------------------------------------------------------------------------
# Layout 1: Strategy Performance -> Trades List (paired entry/exit rows)
# ---------------------------------------------------------------------------
def _parse_ts_performance_trades_list_csv(content: bytes) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
    """
    ``TradeStation Trades List`` block: header ``#,Type,Date/Time,...`` then paired
    rows (entry leg, exit leg) sharing one Date/Time column.

    Returns (handled, trades, error). ``handled=False`` -> not this layout.
    """
    text = _try_decode_csv_text(content)
    if text is None:
        return False, [], None

    marker = "TradeStation Trades List"
    if marker not in text:
        return False, [], None

    chunk_start = text.find(marker)
    lines = text[chunk_start:].splitlines()

    hdr_idx: Optional[int] = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("#,") and "Date/Time" in line:
            hdr_idx = i
            break

    if hdr_idx is None:
        # Title "TradeStation Trades List" without the paired-row header (e.g. the
        # open-profit layout) is a different format - let other parsers try it.
        return False, [], None

    section_lines = [lines[hdr_idx]]
    stop_tokens = frozenset({"Annual", "Monthly", "Weekly", "Quarterly"})
    for line in lines[hdr_idx + 1 :]:
        t = line.strip()
        if not t:
            continue
        if t in stop_tokens:
            break
        if t.startswith("TradeStation Periodical"):
            break
        section_lines.append(_sanitize_ts_trades_list_csv_line(line))

    chunk = "\n".join(section_lines)
    try:
        df = pd.read_csv(io.StringIO(chunk), engine="python")
    except Exception as e:
        return True, [], f"Could not read TradeStation Trades List table: {e}"

    if df.empty:
        return True, [], "TradeStation Trades List table is empty."

    dt_col = None
    for c in df.columns:
        if "date/time" in _norm_header(str(c)):
            dt_col = str(c)
            break
    if dt_col is None:
        return True, [], "Trades list has no Date/Time column."

    pnl_col = None
    for c in df.columns:
        nh = _norm_header(str(c))
        if "cum" in nh and "net profit" in nh:
            pnl_col = str(c)
            break
    if pnl_col is None and df.shape[1] > 7:
        pnl_col = str(df.columns[7])
    if pnl_col is None:
        return True, [], "Could not find Net Profit column in trades list."

    price_col = None
    pl_col = None
    comm_col = None
    slip_col = None
    for c in df.columns:
        nh = _norm_header(str(c))
        if nh == "price":
            price_col = str(c)
        elif "profit" in nh and "loss" in nh and ("shares" in nh or "ctrts" in nh):
            pl_col = str(c)
        elif nh == "comm.":
            comm_col = str(c)
        elif nh == "slippage":
            slip_col = str(c)

    type_col = None
    signal_col = None
    for c in df.columns:
        nh = _norm_header(str(c))
        if nh == "type":
            type_col = str(c)
        elif nh == "signal":
            signal_col = str(c)

    trades: List[Dict[str, Any]] = []
    i = 0
    n = len(df)
    while i < n:
        row = df.iloc[i]
        tid = row.iloc[0]
        if pd.isna(tid) or str(tid).strip() == "":
            i += 1
            continue
        try:
            int(float(str(tid).strip()))
        except (ValueError, TypeError):
            i += 1
            continue

        entry_dt = pd.to_datetime(row[dt_col], errors="coerce", dayfirst=True)
        if i + 1 >= n:
            break
        row2 = df.iloc[i + 1]
        exit_dt = pd.to_datetime(row2[dt_col], errors="coerce", dayfirst=True)
        if pd.isna(entry_dt) or pd.isna(exit_dt):
            i += 1
            continue

        pnl_net = _parse_currency_us(row[pnl_col])
        gross_pnl: Optional[float] = None
        if pl_col is not None and _looks_like_ts_trade_profit_cell(row2.get(pl_col)):
            gross_pnl = float(_parse_currency_us(row2.get(pl_col)))
        trade: Dict[str, Any] = {
            "entry": entry_dt,
            "exit": exit_dt,
            "pnl": float(pnl_net),
        }
        if gross_pnl is not None and abs(gross_pnl - pnl_net) > 1e-6:
            trade["pnl_gross"] = float(gross_pnl)
            trade["pnl_net"] = float(pnl_net)
        if comm_col is not None or slip_col is not None:
            comm = (
                _parse_currency_us(row.get(comm_col)) + _parse_currency_us(row2.get(comm_col))
                if comm_col
                else 0.0
            )
            slip = (
                _parse_currency_us(row.get(slip_col)) + _parse_currency_us(row2.get(slip_col))
                if slip_col
                else 0.0
            )
            if comm or slip:
                trade["ts_commission"] = float(comm)
                trade["ts_slippage"] = float(slip)
        if price_col is not None:
            ep = _parse_ts_price(row.get(price_col))
            xp = _parse_ts_price(row2.get(price_col))
            if ep is not None:
                trade["entry_price"] = float(ep)
            if xp is not None:
                trade["exit_price"] = float(xp)
        if type_col is not None:
            trade["entry_type"] = str(row.get(type_col) or "").strip()
            trade["exit_type"] = str(row2.get(type_col) or "").strip()
        if signal_col is not None:
            trade["entry_signal"] = str(row.get(signal_col) or "").strip()
            trade["exit_signal"] = str(row2.get(signal_col) or "").strip()
        trades.append(trade)
        i += 2

    if not trades:
        return True, [], "No paired entry/exit rows parsed from TradeStation Trades List."

    trades.sort(key=lambda r: r["entry"])
    return True, trades, None


# ---------------------------------------------------------------------------
# Layout 2: MultiWalk walk-forward TradeData.csv (Entry / Exit event rows)
# ---------------------------------------------------------------------------
def _parse_ts_date_hhmm(date_val: Any, time_val: Any) -> Optional[pd.Timestamp]:
    """Parse MultiWalk TradeData ``Date`` (MM/DD/YYYY) + ``Time`` (HHMM int, e.g. 2100 -> 21:00)."""
    dt = pd.to_datetime(date_val, dayfirst=False, errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime(date_val, dayfirst=True, errors="coerce")
    if pd.isna(dt):
        return None
    try:
        t_raw = int(float(str(time_val).strip()))
    except (TypeError, ValueError):
        t_raw = 0
    hours, mins = divmod(t_raw, 100)
    return pd.Timestamp(dt.normalize()) + pd.Timedelta(hours=hours, minutes=mins)


def _parse_multiwalk_wf_trade_data_csv(content: bytes) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
    """
    MultiWalk walk-forward ``TradeData.csv``: one row per Entry/Exit event.

    Columns: ``Date,Time,Price,Type,Position,PNL,MAE,MFE`` (Type = Entry | Exit).
    """
    text = _try_decode_csv_text(content)
    if text is None:
        return False, [], None

    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception:
        return False, [], None

    if df.empty or len(df.columns) < 5:
        return False, [], None

    col_map = {_norm_header(c): str(c) for c in df.columns}
    need = ("date", "time", "type", "position", "pnl")
    if not all(k in col_map for k in need):
        return False, [], None

    type_vals = df[col_map["type"]].astype(str).str.strip().str.lower()
    if not type_vals.isin(("entry", "exit")).any():
        return False, [], None

    price_col = col_map.get("price")
    date_col = col_map["date"]
    time_col = col_map["time"]
    type_col = col_map["type"]
    pos_col = col_map["position"]
    pnl_col = col_map["pnl"]

    trades: List[Dict[str, Any]] = []
    pending: Optional[Dict[str, Any]] = None

    for _, row in df.iterrows():
        kind = str(row.get(type_col) or "").strip().lower()
        if kind not in ("entry", "exit"):
            continue
        pos = str(row.get(pos_col) or "").strip()
        ts = _parse_ts_date_hhmm(row.get(date_col), row.get(time_col))
        if ts is None:
            continue
        px = _parse_ts_price(row.get(price_col)) if price_col else None

        if kind == "entry":
            pending = {"entry": ts, "position": pos}
            if px is not None and abs(px) > 1e-9:
                pending["entry_price"] = px
            continue

        if pending is None:
            continue
        trade: Dict[str, Any] = {
            "entry": pending["entry"],
            "exit": ts,
            "pnl": float(_parse_currency_us(row.get(pnl_col))),
            "entry_type": pending.get("position") or pos,
            "exit_type": pos,
        }
        if pending.get("entry_price") is not None:
            trade["entry_price"] = pending["entry_price"]
        if px is not None and abs(px) > 1e-9:
            trade["exit_price"] = px
        trades.append(trade)
        pending = None

    if not trades:
        return True, [], "MultiWalk TradeData: no Entry/Exit pairs parsed."

    trades.sort(key=lambda r: r["entry"])
    return True, trades, None


# ---------------------------------------------------------------------------
# Layout 3: Flat one-row-per-trade table with Entry/Exit columns + P/L column
# ---------------------------------------------------------------------------
def _combine_split_dt_series(df: pd.DataFrame, label: str) -> Optional[pd.Series]:
    """Build one datetime Series from separate Entry/Exit Date + Time columns."""
    date_cols = []
    time_cols = []
    for c in df.columns:
        n = _norm_header(c)
        if label not in n:
            continue
        if "date" in n and "time" not in n:
            date_cols.append(c)
        elif "time" in n and "date" not in n:
            time_cols.append(c)
    if not date_cols or not time_cols:
        return None
    dcol = sorted(date_cols, key=lambda x: len(_norm_header(x)))[0]
    tcol = sorted(time_cols, key=lambda x: len(_norm_header(x)))[0]
    try:
        combined = pd.to_datetime(
            df[dcol].astype(str).str.strip() + " " + df[tcol].astype(str).str.strip(),
            errors="coerce",
        )
        if combined.notna().sum() >= max(3, int(0.25 * len(df))):
            return combined
    except Exception:
        return None
    return None


def _pick_datetime_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, Optional[str]]:
    """Return (dataframe possibly with synthetic cols, entry_col, exit_col, error)."""
    cols = list(df.columns)
    normed = {_norm_header(c): c for c in cols}

    def find_named(keys: Tuple[str, ...]) -> Optional[str]:
        for k in keys:
            if k in normed:
                return normed[k]
        for c in cols:
            n = _norm_header(c)
            for k in keys:
                if k in n.replace("/", " ").replace("-", " "):
                    return c
        return None

    work = df
    copied = False

    def ensure_copy() -> None:
        nonlocal work, copied
        if not copied:
            work = df.copy()
            copied = True

    e_dt = find_named(("entry date/time", "entry datetime", "entry time", "entry date"))
    x_dt = find_named(("exit date/time", "exit datetime", "exit time", "exit date"))

    if e_dt is None:
        ser_e = _combine_split_dt_series(df, "entry")
        if ser_e is not None:
            ensure_copy()
            e_dt = "__ts_verify_entry_dt__"
            work[e_dt] = ser_e

    if x_dt is None:
        ser_x = _combine_split_dt_series(work, "exit")
        if ser_x is not None:
            ensure_copy()
            x_dt = "__ts_verify_exit_dt__"
            work[x_dt] = ser_x

    if e_dt and x_dt:
        return work, e_dt, x_dt, None

    scores: List[Tuple[float, str]] = []
    for c in list(work.columns):
        parsed = pd.to_datetime(work[c], errors="coerce")
        ratio = float(parsed.notna().mean()) if len(work) else 0.0
        if ratio >= 0.4:
            scores.append((ratio, str(c)))
    scores.sort(reverse=True)
    if len(scores) >= 2:
        c1, c2 = scores[0][1], scores[1][1]
        n1, n2 = _norm_header(c1), _norm_header(c2)
        if "exit" in n1 or "cover" in n1:
            return work, c2, c1, None
        if "exit" in n2 or "cover" in n2:
            return work, c1, c2, None
        return work, c1, c2, None

    return df, "", "", "Could not detect Entry and Exit time columns. Export a trade list with entry/exit times."


def _pick_pnl_column(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    prefer = (
        "net profit",
        "trade profit",
        "tradeprofit",
        "netprofit",
        "$ profit",
        "profit loss",
        "pnl",
        "profit",
    )
    cols = list(df.columns)
    for p in prefer:
        for c in cols:
            if _norm_header(c) == p:
                return str(c), None
    for p in prefer:
        for c in cols:
            n = _norm_header(c)
            if p in n and "price" not in n:
                return str(c), None

    best_c = None
    best_skew = -1.0
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        valid = float(s.notna().mean()) if len(df) else 0.0
        if valid < 0.4:
            continue
        skew = abs(float(s.fillna(0).mean()))
        if skew > best_skew:
            best_skew = skew
            best_c = c
    if best_c:
        return str(best_c), None
    return "", "Could not detect a numeric profit column (e.g. Net Profit)."


def _parse_flat_trade_table(content: bytes, filename: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        df = _read_export_table(content, filename)
    except Exception as e:
        return [], str(e)

    if df is None or df.empty:
        return [], "Empty spreadsheet."

    df = df.dropna(axis=0, how="all")
    df_work, entry_col, exit_col, err_dt = _pick_datetime_columns(df)
    if err_dt:
        return [], err_dt
    pnl_col, err_pnl = _pick_pnl_column(df_work)
    if err_pnl:
        return [], err_pnl

    trades: List[Dict[str, Any]] = []
    for _, row in df_work.iterrows():
        et = pd.to_datetime(row.get(entry_col), errors="coerce")
        xt = pd.to_datetime(row.get(exit_col), errors="coerce")
        pnl = pd.to_numeric(row.get(pnl_col), errors="coerce")
        if pd.isna(et) or pd.isna(xt):
            continue
        if pd.isna(pnl):
            pnl = 0.0
        trades.append({"entry": et, "exit": xt, "pnl": float(pnl)})

    if not trades:
        return [], "No rows with valid entry time, exit time, and profit."

    trades.sort(key=lambda r: r["entry"])
    return trades, None


# ---------------------------------------------------------------------------
# Layout 4: TradeStation Trades List with daily open profit
# ---------------------------------------------------------------------------
def _parse_open_profit_csv(content: bytes) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
    """
    ``Trade Number, Type, Date, open profit, Price`` with daily open-profit rows
    between the entry row (Sell Short / Buy) and the exit row (Buy to Cover / Sell).
    """
    text = _try_decode_csv_text(content)
    if text is None:
        return False, [], None
    if "open profit" not in text:
        return False, [], None

    try:
        df = pd.read_csv(io.StringIO(text), skiprows=2, engine="python")
    except Exception:
        return False, [], None
    df.columns = df.columns.str.strip()

    trades: Dict[int, Dict[str, Any]] = {}
    current: Optional[Dict[str, Any]] = None
    current_num: Optional[int] = None

    for _, row in df.iterrows():
        if pd.isna(row.get("Trade Number", "")) and pd.isna(row.get("Type", "")):
            continue
        trade_num = row.get("Trade Number", "")
        trade_type = str(row.get("Type", "")).strip()
        date_str = str(row.get("Date", "")).strip()
        open_profit = row.get("open profit", "")
        price = row.get("Price", "")

        if not pd.isna(trade_num) and str(trade_num).strip() != "":
            try:
                current_num = int(float(str(trade_num)))
                current = {
                    "entry": None,
                    "exit": None,
                    "entry_price": None,
                    "exit_price": None,
                    "daily_profits": [],
                }
                trades[current_num] = current
            except (ValueError, TypeError):
                continue

        if not date_str or date_str == "nan":
            continue
        date = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
        if pd.isna(date):
            continue

        is_entry = "sell short" in trade_type.lower() or "buy" in trade_type.lower()
        is_exit = "buy to cover" in trade_type.lower() or (
            "sell" in trade_type.lower() and "short" not in trade_type.lower()
        )

        if current is None:
            continue
        if is_entry and current["entry"] is None and not pd.isna(price) and str(price).strip():
            current["entry"] = date
            current["entry_price"] = _parse_ts_price(price)
        elif is_exit and current["exit"] is None and not pd.isna(price) and str(price).strip():
            current["exit"] = date
            current["exit_price"] = _parse_ts_price(price)
        elif current["exit"] is None and not pd.isna(open_profit) and str(open_profit).strip():
            try:
                current["daily_profits"].append((date, float(open_profit)))
            except (ValueError, TypeError):
                pass

    out: List[Dict[str, Any]] = []
    for num, t in trades.items():
        if t["entry"] is None or t["exit"] is None:
            continue
        pnl = 0.0
        if t["entry_price"] is not None and t["exit_price"] is not None:
            pnl = t["exit_price"] - t["entry_price"]
        elif t["daily_profits"]:
            pnl = t["daily_profits"][-1][1]
        trade: Dict[str, Any] = {"entry": t["entry"], "exit": t["exit"], "pnl": float(pnl)}
        if t["entry_price"] is not None:
            trade["entry_price"] = t["entry_price"]
        if t["exit_price"] is not None:
            trade["exit_price"] = t["exit_price"]
        if t["daily_profits"]:
            trade["daily_profits"] = t["daily_profits"]
        out.append(trade)

    if not out:
        return True, [], "open profit CSV found but no complete entry/exit trades parsed."
    out.sort(key=lambda r: r["entry"])
    return True, out, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_tradestation_trade_export(
    content: bytes,
    filename: str,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Parse a TradeStation export into ``(trades, error)``.

    Each trade: ``{"entry": Timestamp, "exit": Timestamp, "pnl": float, ...}``.
    Returns ``([], error)`` on failure.
    """
    fn = (filename or "").lower()
    if fn.endswith((".csv", ".xlsx")) or fn.endswith(".csv"):
        handled, tl_trades, tl_err = _parse_ts_performance_trades_list_csv(content)
        if handled:
            return tl_trades, tl_err
        handled, mw_trades, mw_err = _parse_multiwalk_wf_trade_data_csv(content)
        if handled:
            return mw_trades, mw_err
        handled, op_trades, op_err = _parse_open_profit_csv(content)
        if handled:
            return op_trades, op_err

    trades, err = _parse_flat_trade_table(content, filename)
    if err:
        return [], err
    return trades, None
