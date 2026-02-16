# Credit Portfolio AI Analyst (Vercel Edition)

Business Intelligence MVP that explains changes in portfolio credit risk between two reporting dates.

## Stack
- Python
- SQLite (in-memory)
- Pandas
- OpenAI API compatible clients (OpenAI + Google Gemini endpoint)
- FastAPI (serverless API for Vercel)
- Static HTML/CSS/JavaScript frontend

## Architecture
- `index.html`, `styles.css`, `script.js`:
  Browser UI with:
  - left-side filters
  - group-by and aggregation controls
  - asset-class comparison table (2 selected dates as columns)
  - step-by-step analysis output
  - execution window (logs, compiled SQL, execution results)
- `api/index.py`:
  Vercel Python API endpoint for:
  - model discovery
  - data retrieval
  - analysis execution
- `credit_engine.py`:
  Shared analysis engine (data generation, SQL safety, reasoning loop, drill-down logic)
- `vercel.json`:
  Rewrites `/api/*` to `/api/index.py`

## API Routes
- `GET /api/health`
- `GET /api/data`
- `POST /api/models`
- `POST /api/analyze`

## Local Development
Install dependencies:

```bash
uv sync
```

Run API locally:

```bash
uv run uvicorn api.index:app --reload --port 8000
```

Serve frontend locally (second terminal):

```bash
python -m http.server 3000
```

Open:
- UI: `http://localhost:3000`
- API: `http://localhost:8000/api/health`

## Deploy to Vercel
1. Push this branch to GitHub.
2. In Vercel dashboard, click **Add New Project**.
3. Import `magpieprojects/robo-credit-analyst`.
4. Keep root directory as repository root.
5. Deploy (preview deployment first).
6. Open preview URL and test:
   - load models with API key
   - run analysis
   - verify execution window and drill-down output
7. Promote to production when ready.

## Notes
- API keys are entered at runtime by users in the browser UI.
- Do not commit real API keys.
- SQL from LLM output is validated to single `SELECT` statements against `risk_data`.
- Synthetic data includes asset classes:
  - `stock`, `bond`, `real estate`, `private equity`, `hedge fund`
