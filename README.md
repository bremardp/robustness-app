# TradeStation Robustness Analyzer

A standalone Docker app that measures the **robustness of a trading strategy** from
a TradeStation **trades list CSV**. No OHLC price data needed — everything runs off
the closed-trade P/L list you already have in the export.

> **Fully standalone** — there are **no dependencies on the LEGO Block platform**.
> Everything (CSV parser, metrics, Monte Carlo engine, web UI) is self-contained in
> this repository. Anyone with Docker (or Python) can run it on their own machine.

## Quick start

```bash
cd robustness-app
docker compose up --build
```

Open <http://localhost:8001>, upload a TradeStation trades CSV, click **Analyze
robustness**. Or call the API directly:

```bash
curl -F "file=@sample_data/sample_trades_list.csv" -F "n_sims=2000" \
     -F "starting_equity=100000" http://localhost:8001/api/analyze
```

## Run without Docker (plain Python)

```bash
cd robustness-app
python -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

The Chart.js library used by the web UI is vendored in `app/static/` — no internet
access or CDN is required.

## What it does

1. **Parses** the CSV (auto-detects the layout, see below) into a uniform trade list.
2. **Summarizes** the strategy: net/gross P/L, profit factor, win rate, avg win/loss,
   expectancy, max drawdown, recovery factor, streaks.
3. **Runs 9 robustness tests** on the trade list and grades each one (pass / warn / fail):

| Test | Question it answers |
|---|---|
| Minimum trades | Is the sample large enough for meaningful statistics? |
| Monte Carlo reshuffle | Does the result depend on a lucky trade order? |
| Monte Carlo bootstrap | What is ruin probability and median drawdown if you re-draw trades? |
| Randomized trade skip | Does the edge survive dropping random trades? |
| Profit factor stability | Does the profit factor stay above 1.0 when re-sampled? |
| Drawdown consistency | Is the realized drawdown unusual vs random trade order? |
| Bootstrap mean CI | Is the average trade reliably positive (95% CI)? |
| Risk of ruin | Probability of touching the ruin floor over a trade run |
| R-Squared slope equity curve | Does equity grow along a straight-line trend (R² of OLS fit vs trade index)? |

4. Produces an **overall robustness score (0–100)** and verdict, with equity curve,
   Monte Carlo percentile band, drawdown, and trade P/L histogram charts.

## Supported TradeStation CSV layouts (auto-detected)

- **Strategy Performance → Trades List** (paired entry/exit rows, `#,Type,Date/Time,...,Net Profit`).
- **MultiWalk walk-forward `TradeData.csv`** (`Date,Time,Price,Type,Position,PNL,MAE,MFE`).
- **Flat one-row-per-trade table** with Entry/Exit date-time columns and a P/L column.
- **Trades List with open profit** (`Trade Number,Type,Date,open profit,Price`).

## API

| Endpoint | Description |
|---|---|
| `POST /api/analyze` | multipart: `file` (required), `n_sims` (50–20000), `seed`, `starting_equity`, `ruin_point`, `skip_fraction` |
| `GET /` | Web UI |
| `GET /docs` | Swagger UI |
| `GET /health` | Liveness probe |

The response is a JSON report: `summary` (metrics), `robustness` (per-test results
+ score + verdict), `series` (equity/drawdown/P&L arrays for charts), `trades`
(parsed list), `settings`.

## Configuration

- Port is mapped to `8001` in `docker-compose.yml` (change freely).
- `MAX_UPLOAD_MB` env var caps upload size (default 50 MB).
- If your corporate proxy breaks pip TLS during the Docker build, the image already
  passes `--trusted-host pypi.org --trusted-host files.pythonhosted.org`.

## Development

```bash
python -m pip install -r requirements-dev.txt
pytest tests -q          # unit + API tests
```

Sample files in `sample_data/` (regenerate with `py sample_data/generate_samples.py`).
