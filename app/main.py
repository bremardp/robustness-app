"""
TradeStation CSV robustness analyzer - FastAPI application.

POST /api/analyze  multipart form:  file=..., n_sims=..., seed=..., starting_equity=...
                     ruin_point=..., skip_fraction=...
GET  /health       liveness probe
GET  /             minimal HTML/JS UI
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.report import build_report

app = FastAPI(title="TradeStation Robustness Analyzer", version="1.0.0")

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024

app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    ui = _STATIC_DIR / "index.html"
    if not ui.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return ui.read_text(encoding="utf-8")


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    n_sims: Optional[int] = Form(2000),
    seed: Optional[int] = Form(None),
    starting_equity: Optional[float] = Form(100_000.0),
    ruin_point: Optional[float] = Form(None),
    skip_fraction: Optional[float] = Form(0.10),
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {_MAX_BYTES // (1024 * 1024)} MB).",
        )

    n_sims_val = max(50, min(int(n_sims or 2000), 20_000))
    start_eq = float(starting_equity or 100_000.0)
    skip_frac = 0.10 if skip_fraction is None else min(0.5, max(0.01, float(skip_fraction)))

    report = build_report(
        content,
        file.filename,
        n_sims=n_sims_val,
        seed=seed,
        starting_equity=start_eq,
        ruin_point=None if ruin_point is None else float(ruin_point),
        skip_fraction=skip_frac,
    )
    if not report.get("ok"):
        raise HTTPException(status_code=422, detail=report.get("error", "Could not analyze file."))
    return report
