import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from credit_engine import (
    ASSET_CLASSES,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    GEMINI_PROVIDER,
    OPENAI_PROVIDER,
    discover_available_models,
    load_risk_data,
    run_analysis,
    serialize_analysis_result,
)

app = FastAPI(title="Credit Portfolio AI Analyst API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _static_file_response(filename: str) -> FileResponse:
    path = PROJECT_ROOT / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Static file not found: {filename}")
    return FileResponse(path)


def _api_root_payload() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Credit Portfolio AI Analyst API is running.",
        "ui": "/",
        "health": "/api/health",
    }


class AnalyzeRequest(BaseModel):
    model: str = Field(min_length=1)
    seed: int = 42


def _normalize_provider(provider: str) -> str:
    cleaned = provider.strip()
    if cleaned in {OPENAI_PROVIDER, GEMINI_PROVIDER}:
        return cleaned
    raise ValueError(f"Unsupported provider: {provider}")


def _server_provider() -> str:
    configured = os.getenv("LLM_PROVIDER", OPENAI_PROVIDER)
    return _normalize_provider(configured)


def _server_api_key(provider: str) -> str:
    if provider == GEMINI_PROVIDER:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
    else:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")

    if not api_key:
        if provider == GEMINI_PROVIDER:
            raise ValueError("Missing GEMINI_API_KEY (or fallback LLM_API_KEY) environment variable.")
        raise ValueError("Missing OPENAI_API_KEY (or fallback LLM_API_KEY) environment variable.")
    return api_key


@app.get("/health")
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> FileResponse:
    return _static_file_response("index.html")


@app.get("/index.html")
def index_page() -> FileResponse:
    return _static_file_response("index.html")


@app.get("/script.js")
def script_asset() -> FileResponse:
    return _static_file_response("script.js")


@app.get("/styles.css")
def styles_asset() -> FileResponse:
    return _static_file_response("styles.css")


@app.get("/api")
@app.get("/api/")
@app.get("/api/index.py")
def api_root() -> dict[str, str]:
    return _api_root_payload()


@app.get("/data")
@app.get("/api/data")
def data(seed: int = 42) -> dict[str, Any]:
    rows = load_risk_data(seed=seed)
    reporting_dates = sorted({str(row["reporting_date"]) for row in rows})
    ratings = sorted({str(row["rating"]) for row in rows})
    asset_classes = [item for item in ASSET_CLASSES if item in {str(row["asset_class"]) for row in rows}]
    return {
        "rows": rows,
        "metadata": {
            "reporting_dates": reporting_dates,
            "ratings": ratings,
            "asset_classes": asset_classes,
        },
    }


@app.get("/models")
@app.get("/api/models")
def models() -> dict[str, Any]:
    try:
        provider = _server_provider()
        api_key = _server_api_key(provider)
        model_ids = discover_available_models(
            api_key=api_key,
            provider=provider,
        )
        default_model = (
            DEFAULT_OPENAI_MODEL if provider == OPENAI_PROVIDER else DEFAULT_GEMINI_MODEL
        )
        selected_default = default_model if default_model in model_ids else model_ids[0]
        return {"provider": provider, "models": model_ids, "default_model": selected_default}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/analyze")
@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
    try:
        provider = _server_provider()
        api_key = _server_api_key(provider)
        result = run_analysis(
            provider=provider,
            api_key=api_key,
            model=payload.model.strip(),
            seed=payload.seed,
        )
        return serialize_analysis_result(result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
