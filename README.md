# Credit Portfolio AI Analyst (Vercel Deployment)

Business Intelligence MVP that explains why portfolio credit risk changed between two reporting dates (`2024-01-31` and `2024-02-29`).

## Current Architecture
- Static web UI:
  - `index.html`
  - `script.js`
  - `styles.css`
- Serverless API (Vercel Python):
  - `api/index.py`
- Shared analysis engine:
  - `credit_engine.py`
- Vercel routing config:
  - `vercel.json`

Legacy Streamlit UI was removed from this branch to keep Vercel packaging minimal.

## What The App Does
- Loads deterministic synthetic credit data (100 rows).
- Includes `asset_class` values:
  - `stock`, `bond`, `real estate`, `private equity`, `hedge fund`
- Provides left-side filters, group-by, and aggregations.
- Shows asset-class comparison with exposure and average PD for 2 selected dates.
- Runs a strict 3-step LLM analysis:
  1. Portfolio weighted average PD trigger
  2. Customer attribution drill-down
  3. Root cause and 10-word summary
- Displays execution logs, compiled SQL, query results, drill-down output, and final note.

## API Endpoints
- `GET /api/health`
- `GET /api/data`
- `GET /api/models`
- `POST /api/analyze`

## Runtime Dependencies (Deploy)
Defined in `requirements.txt`:
- `fastapi`
- `openai`

This dependency set is intentionally minimal for Vercel function size constraints.

Deployment packaging is further reduced via `.vercelignore` (for example excluding `app.py`, `.venv`, and `uv.lock`).

## Local Development
1. Install dependencies:
```bash
uv sync
```

2. Install local dev extra (for `uvicorn`):
```bash
uv sync --extra dev
```

3. Create `.env` in repo root (example below):
```bash
LLM_PROVIDER="Azure OpenAI"
AZURE_OPENAI_API_KEY="key"
AZURE_OPENAI_ENDPOINT="https://fndry-mp-ai-dev-gen-gwc.cognitiveservices.azure.com/"
AZURE_OPENAI_API_VERSION="2024-12-01-preview"
AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4.1"
```

4. Start API (loads `.env` automatically):
```bash
uv run --env-file .env uvicorn api.index:app --reload --port 8000
```

5. Start static frontend in another terminal:
```bash
python -m http.server 3000
```

6. Open:
- `http://localhost:3000` (UI)
- `http://localhost:8000/api/health` (API health)

`script.js` automatically uses `http://localhost:8000` when frontend runs on port `3000`.

## Deploy To Vercel
1. Push your branch to GitHub.
2. In Vercel, choose `Add New` -> `Project`.
3. Import existing repository `magpieprojects/robo-credit-analyst`.
4. Use root directory `.`.
5. Keep `vercel.json` in repo root.
6. Configure environment variables in Vercel:
   - `LLM_PROVIDER` = `OpenAI`, `Google Gemini`, or `Azure OpenAI` (default is `OpenAI`)
   - `OPENAI_API_KEY` for OpenAI mode
   - `GEMINI_API_KEY` for Gemini mode
   - `AZURE_OPENAI_API_KEY` for Azure OpenAI mode
   - `AZURE_OPENAI_ENDPOINT` for Azure OpenAI mode
   - `AZURE_OPENAI_API_VERSION` for Azure OpenAI mode
   - `AZURE_OPENAI_CHAT_DEPLOYMENT` for Azure OpenAI mode (deployment name used for chat)
   - optional fallback: `LLM_API_KEY`
   - optional request timeout in seconds: `LLM_TIMEOUT_SECONDS` (default `15`)
   - optional retry count: `LLM_MAX_RETRIES` (default `0`)
7. Deploy preview first.
8. Promote to production when checks pass.

If Vercel asks for a repository name that already exists, switch to importing the existing Git repository instead of creating a new one.

## Troubleshooting: 504 `FUNCTION_INVOCATION_TIMEOUT`
- Symptom example:
  - `Analysis failed (504): ... FUNCTION_INVOCATION_TIMEOUT ...`
- Cause:
  - `/api/analyze` performs multiple sequential LLM calls; slow model/provider responses can exceed function limits.
- Fixes already in this repo:
  - LLM client calls respect `LLM_TIMEOUT_SECONDS` and `LLM_MAX_RETRIES`.
- Vercel runtime duration:
  - Configure max duration in Vercel Project Settings (Functions), not `vercel.json` `functions`,
    if you see pattern match errors like:
    - `The pattern "..." defined in functions doesn't match any Serverless Functions...`
- Recommended Vercel settings:
  - Function max duration: `60` seconds (or higher if your plan allows)
  - Keep `LLM_TIMEOUT_SECONDS=15`
  - Keep `LLM_MAX_RETRIES=0`
  - Use a lower-latency model for analysis when possible.

## Post-Deploy Checks
1. `GET /api/health` returns `{"status":"ok"}`.
2. `GET /api/data` returns 100 rows and asset classes.
3. UI loads models from server-managed credentials (no user API key input).
4. Analysis run shows Step 1, Step 2, and Step 3 panels.
5. Analysis run shows Execution Window tabs (Log Output, Compiled SQL, Execution Results).
6. Analysis run shows drill-down output and final note.

## Security Notes
- API keys are server-managed via environment variables.
- UI does not accept or store user API keys.
- Do not commit API keys or `.env` files.
- Keep local secrets out of git (`.streamlit/secrets.toml`, `.env`, `.venv` are ignored).
