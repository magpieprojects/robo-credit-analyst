import random
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

import pandas as pd
from openai import OpenAI

T0_DATE = "2024-01-31"
T1_DATE = "2024-02-29"
BIGCORP_NAME = "BigCorp Inc"
ASSET_CLASSES = ["stock", "bond", "real estate", "private equity", "hedge fund"]

SCHEMA_METADATA = [
    ("risk_data", "table", "Contains historical credit risk data for customers.", ""),
    ("customer_id", "column", "Unique identifier for the customer.", ""),
    (
        "rating",
        "column",
        "Credit rating of the customer. AAA is best, CCC is worst.",
        "AAA, AA, A, BBB, BB, B, CCC",
    ),
    ("pd", "column", "Probability of Default (0.0 to 1.0).", "0.0 - 1.0"),
    ("exposure", "column", "Total credit exposure (EAD) in EUR.", ""),
    (
        "asset_class",
        "column",
        "Asset class of the credit exposure.",
        "stock, bond, real estate, private equity, hedge fund",
    ),
    (
        "reporting_date",
        "column",
        "The date the data was recorded (YYYY-MM-DD).",
        "Quarterly dates",
    ),
]

RATING_SCALE = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
RATING_TO_PD = {
    "AAA": 0.001,
    "AA": 0.003,
    "A": 0.010,
    "BBB": 0.020,
    "BB": 0.050,
    "B": 0.100,
    "CCC": 0.200,
}

BANNED_SQL_TOKENS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "pragma",
    "attach",
    "detach",
)

OPENAI_PROVIDER = "OpenAI"
GEMINI_PROVIDER = "Google Gemini"
OPENAI_MODEL_PREFIX_ALLOWLIST = ("gpt-",)
GEMINI_MODEL_PREFIX_ALLOWLIST = ("gemini-",)
DEFAULT_OPENAI_MODEL = "gpt-4"
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _validate_sql(sql: str) -> str:
    cleaned = _strip_markdown_fences(sql)
    if not cleaned:
        raise ValueError("LLM returned an empty SQL statement.")

    statement = cleaned.strip()
    if statement.endswith(";"):
        statement = statement[:-1].strip()

    if ";" in statement:
        raise ValueError("Only single SQL statements are allowed.")

    lower_sql = statement.lower()
    if not lower_sql.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    for token in BANNED_SQL_TOKENS:
        if re.search(rf"\b{token}\b", lower_sql):
            raise ValueError(f"SQL contains forbidden token: {token}")

    if "risk_data" not in lower_sql:
        raise ValueError("SQL must reference the risk_data table.")

    return statement


def _word_count(text: str) -> int:
    return len([part for part in text.split() if part.strip()])


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def _compile_param_sql(sql: str, params: list[Any]) -> str:
    normalized_sql = "\n".join(line.rstrip() for line in sql.strip().splitlines())
    parts = normalized_sql.split("?")
    if len(parts) - 1 != len(params):
        return normalized_sql

    rendered_parts: list[str] = [parts[0]]
    for index, param in enumerate(params):
        rendered_parts.append(_sql_literal(param))
        rendered_parts.append(parts[index + 1])
    return "".join(rendered_parts).strip()


def _build_openai_client(api_key: str, provider: str) -> OpenAI:
    if provider == GEMINI_PROVIDER:
        return OpenAI(api_key=api_key, base_url=GEMINI_OPENAI_BASE_URL)
    return OpenAI(api_key=api_key)


def _extract_model_ids(model_list_response: Any) -> list[str]:
    raw_model_ids: list[str] = []
    for model_obj in getattr(model_list_response, "data", []):
        model_id = getattr(model_obj, "id", None)
        if isinstance(model_id, str) and model_id.strip():
            raw_model_ids.append(model_id.strip())
    return sorted(set(raw_model_ids))


def _probe_chat_model(client: OpenAI, model_id: str) -> bool:
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Reply with OK."}],
            temperature=0,
        )
        return bool(getattr(response, "choices", []))
    except Exception:
        return False


def discover_available_models(api_key: str, provider: str) -> list[str]:
    client = _build_openai_client(api_key=api_key, provider=provider)

    if provider == OPENAI_PROVIDER:
        response = client.models.list()
        unique_model_ids = _extract_model_ids(response)
        valid_models = [
            model_id
            for model_id in unique_model_ids
            if model_id.startswith(OPENAI_MODEL_PREFIX_ALLOWLIST)
        ]
        if not valid_models:
            raise ValueError(
                "No valid GPT models were found for this API key. "
                "Please verify model access in your OpenAI project."
            )
        valid_models.sort(
            key=lambda model_id: (0 if model_id == DEFAULT_OPENAI_MODEL else 1, model_id)
        )
        return valid_models

    discovered_model_ids: list[str] = []
    try:
        response = client.models.list()
        discovered_model_ids = [
            model_id
            for model_id in _extract_model_ids(response)
            if model_id.startswith(GEMINI_MODEL_PREFIX_ALLOWLIST)
        ]
    except Exception:
        discovered_model_ids = []

    if DEFAULT_GEMINI_MODEL not in discovered_model_ids:
        discovered_model_ids.insert(0, DEFAULT_GEMINI_MODEL)

    valid_models = [
        model_id for model_id in discovered_model_ids if _probe_chat_model(client, model_id)
    ]
    if not valid_models:
        raise ValueError(
            "No valid Gemini models were found for this API key. "
            "Please verify model access and that gemini-flash-latest is enabled."
        )

    valid_models.sort(
        key=lambda model_id: (0 if model_id == DEFAULT_GEMINI_MODEL else 1, model_id)
    )
    return valid_models


def _schema_to_prompt_block(schema_metadata: list[tuple[str, str, str, str]]) -> str:
    lines = []
    for name, item_type, description, allowed_values in schema_metadata:
        if item_type == "table":
            lines.append(f"- Table `{name}`: {description}")
        else:
            values = allowed_values if allowed_values else "N/A"
            lines.append(f"- Column `{name}`: {description} Allowed values: {values}")
    return "\n".join(lines)


@dataclass
class OpenAIReasoner:
    api_key: str
    model: str = DEFAULT_OPENAI_MODEL
    temperature: float = 0.0
    provider: str = OPENAI_PROVIDER

    def __post_init__(self) -> None:
        self.client = _build_openai_client(api_key=self.api_key, provider=self.provider)

    def _call(self, prompt: str) -> str:
        if self.provider == GEMINI_PROVIDER:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            choices = getattr(response, "choices", [])
            if not choices:
                raise ValueError("Google Gemini returned no response choices.")

            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
            raise ValueError("Google Gemini returned no readable text output.")

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=self.temperature,
        )
        text = getattr(response, "output_text", "")
        if text:
            return text.strip()

        chunks: list[str] = []
        for output in getattr(response, "output", []):
            for content in getattr(output, "content", []):
                maybe_text = getattr(content, "text", None)
                if maybe_text:
                    chunks.append(maybe_text)
        combined = "\n".join(chunks).strip()
        if not combined:
            raise ValueError("OpenAI returned no readable text output.")
        return combined

    def generate_portfolio_sql(
        self,
        schema_metadata: list[tuple[str, str, str, str]],
        t0_date: str,
        t1_date: str,
    ) -> str:
        prompt = f"""
Task: Portfolio SQL
You are an expert SQLite analyst.
Return only one valid SQLite SELECT statement. No markdown, no explanation.

Schema metadata:
{_schema_to_prompt_block(schema_metadata)}

Objective:
Compute weighted average PD per reporting_date for {t0_date} and {t1_date}.
Formula: SUM(pd * exposure) / SUM(exposure)

Output columns required:
- reporting_date
- wapd
""".strip()
        return self._call(prompt)

    def generate_customer_drill_down_sql(
        self,
        schema_metadata: list[tuple[str, str, str, str]],
        t0_date: str,
        t1_date: str,
    ) -> str:
        prompt = f"""
Task: Customer Drill Down SQL
You are an expert SQLite analyst.
Return only one valid SQLite SELECT statement. No markdown, no explanation.

Schema metadata:
{_schema_to_prompt_block(schema_metadata)}

Objective:
Return customer-level fields needed for contribution analysis for dates {t0_date} and {t1_date}.

Output columns required:
- customer_id
- reporting_date
- pd
- exposure
""".strip()
        return self._call(prompt)

    def generate_summary_sentence(
        self,
        customer_name: str,
        old_rating: str,
        new_rating: str,
        old_pd: float,
        new_pd: float,
        old_exposure: float,
        new_exposure: float,
        correction: str = "",
    ) -> str:
        prompt = f"""
Task: Summary Sentence
Return exactly 10 words. No bullet points. No explanation.

Facts:
- Customer: {customer_name}
- Rating: {old_rating} -> {new_rating}
- PD: {old_pd:.4f} -> {new_pd:.4f}
- Exposure: {old_exposure:,.2f} -> {new_exposure:,.2f}

Write one sentence explaining why portfolio PD changed.
{correction}
""".strip()
        return self._call(prompt)


class CreditAnalystAgent:
    def __init__(
        self,
        connection: sqlite3.Connection,
        reasoner: OpenAIReasoner,
        schema_metadata: list[tuple[str, str, str, str]],
        t0_date: str = T0_DATE,
        t1_date: str = T1_DATE,
    ) -> None:
        self.connection = connection
        self.reasoner = reasoner
        self.schema_metadata = schema_metadata
        self.t0_date = t0_date
        self.t1_date = t1_date
        self.logs: list[str] = []

    def _log(self, message: str) -> None:
        self.logs.append(message)

    def _run_sql(self, raw_sql: str) -> pd.DataFrame:
        safe_sql = _validate_sql(raw_sql)
        return pd.read_sql_query(safe_sql, self.connection)

    def _step1_portfolio_trigger(self) -> dict[str, Any]:
        self._log("[Step 1] Analyzing Overall Portfolio...")
        sql = self.reasoner.generate_portfolio_sql(
            self.schema_metadata, self.t0_date, self.t1_date
        )
        df = self._run_sql(sql)
        if df.empty or "reporting_date" not in df.columns or "wapd" not in df.columns:
            raise ValueError("Portfolio SQL did not return required columns.")

        wapd_map = {
            str(row["reporting_date"]): float(row["wapd"])
            for _, row in df.iterrows()
            if pd.notna(row["wapd"])
        }
        if self.t0_date not in wapd_map or self.t1_date not in wapd_map:
            raise ValueError("Portfolio SQL did not return both reporting dates.")

        wapd_t0 = wapd_map[self.t0_date]
        wapd_t1 = wapd_map[self.t1_date]
        diff = wapd_t1 - wapd_t0
        self._log(f"   -> Found deviation of {diff:+.2%}")

        return {
            "sql": _validate_sql(sql),
            "wapd_t0": wapd_t0,
            "wapd_t1": wapd_t1,
            "diff": diff,
            "raw_df": df,
        }

    def _step2_customer_attribution(self) -> dict[str, Any]:
        self._log("[Step 2] Drilling down to Counterparty level...")
        sql = self.reasoner.generate_customer_drill_down_sql(
            self.schema_metadata, self.t0_date, self.t1_date
        )
        df = self._run_sql(sql)

        required_columns = {"customer_id", "reporting_date", "pd", "exposure"}
        if not required_columns.issubset(set(df.columns)):
            raise ValueError("Drill-down SQL did not return required columns.")

        df = df[df["reporting_date"].isin([self.t0_date, self.t1_date])].copy()
        if df.empty:
            raise ValueError("Drill-down SQL returned no rows for T0/T1.")

        pivot = df.pivot_table(
            index="customer_id",
            columns="reporting_date",
            values=["pd", "exposure"],
            aggfunc="first",
        )
        required_multi_columns = {
            ("pd", self.t0_date),
            ("pd", self.t1_date),
            ("exposure", self.t0_date),
            ("exposure", self.t1_date),
        }
        if not required_multi_columns.issubset(set(pivot.columns)):
            raise ValueError("Cannot compute attribution because some date columns are missing.")

        attribution_df = pd.DataFrame(
            {
                "customer_id": pivot.index,
                "pd_t0": pivot[("pd", self.t0_date)].astype(float),
                "pd_t1": pivot[("pd", self.t1_date)].astype(float),
                "exposure_t0": pivot[("exposure", self.t0_date)].astype(float),
                "exposure_t1": pivot[("exposure", self.t1_date)].astype(float),
            }
        ).reset_index(drop=True)

        total_exposure_t0 = float(attribution_df["exposure_t0"].sum())
        total_exposure_t1 = float(attribution_df["exposure_t1"].sum())
        if total_exposure_t0 == 0 or total_exposure_t1 == 0:
            raise ValueError("Total exposure is zero; cannot compute weighted contribution.")

        attribution_df["term_t0"] = (
            attribution_df["pd_t0"] * attribution_df["exposure_t0"] / total_exposure_t0
        )
        attribution_df["term_t1"] = (
            attribution_df["pd_t1"] * attribution_df["exposure_t1"] / total_exposure_t1
        )
        attribution_df["contribution"] = attribution_df["term_t1"] - attribution_df["term_t0"]
        attribution_df["abs_contribution"] = attribution_df["contribution"].abs()
        attribution_df = attribution_df.sort_values(
            by="abs_contribution", ascending=False
        ).reset_index(drop=True)

        top_customer = str(attribution_df.loc[0, "customer_id"])
        self._log(f"   -> Identified '{top_customer}' as primary driver.")

        return {
            "sql": _validate_sql(sql),
            "attribution_df": attribution_df,
            "raw_df": df,
            "top_customer": top_customer,
            "contribution_sum": float(attribution_df["contribution"].sum()),
        }

    def _step3_root_cause_summary(self, top_customer: str) -> dict[str, Any]:
        self._log("[Step 3] Analyzing Root Cause...")
        details_sql = """
            SELECT customer_id, rating, pd, exposure, asset_class, reporting_date
            FROM risk_data
            WHERE customer_id = ?
              AND reporting_date IN (?, ?)
            ORDER BY reporting_date
        """
        details_params = [top_customer, self.t0_date, self.t1_date]
        details_df = pd.read_sql_query(details_sql, self.connection, params=details_params)
        if len(details_df) != 2:
            raise ValueError("Top mover details are incomplete for T0/T1.")

        t0_row = details_df[details_df["reporting_date"] == self.t0_date].iloc[0]
        t1_row = details_df[details_df["reporting_date"] == self.t1_date].iloc[0]

        old_rating = str(t0_row["rating"])
        new_rating = str(t1_row["rating"])
        old_pd = float(t0_row["pd"])
        new_pd = float(t1_row["pd"])
        old_exposure = float(t0_row["exposure"])
        new_exposure = float(t1_row["exposure"])

        rating_note = "No rating change"
        if old_rating != new_rating:
            direction = (
                "Downgrade"
                if RATING_SCALE.index(new_rating) > RATING_SCALE.index(old_rating)
                else "Upgrade"
            )
            rating_note = f"Rating {direction} ({old_rating} -> {new_rating})"
            self._log(f"   -> Detected Rating {direction} ({old_rating} -> {new_rating}).")
        else:
            self._log("   -> No rating change detected; checking PD/exposure shifts.")

        summary = self.reasoner.generate_summary_sentence(
            customer_name=top_customer,
            old_rating=old_rating,
            new_rating=new_rating,
            old_pd=old_pd,
            new_pd=new_pd,
            old_exposure=old_exposure,
            new_exposure=new_exposure,
        )
        if _word_count(summary) != 10:
            summary = self.reasoner.generate_summary_sentence(
                customer_name=top_customer,
                old_rating=old_rating,
                new_rating=new_rating,
                old_pd=old_pd,
                new_pd=new_pd,
                old_exposure=old_exposure,
                new_exposure=new_exposure,
                correction=(
                    f"Your previous answer had {_word_count(summary)} words. "
                    "Return exactly 10 words now."
                ),
            )

        if _word_count(summary) != 10:
            words = summary.split()
            if len(words) > 10:
                summary = " ".join(words[:10])
            else:
                summary = " ".join(words + ["today"] * (10 - len(words)))

        self._log(f'[Final Report] "{summary}"')

        return {
            "summary": summary,
            "details_sql": _compile_param_sql(details_sql, details_params),
            "details_df": details_df,
            "rating_change": rating_note,
            "pd_change": (old_pd, new_pd),
            "exposure_change": (old_exposure, new_exposure),
        }

    def run(self) -> dict[str, Any]:
        step1 = self._step1_portfolio_trigger()
        result: dict[str, Any] = {"logs": self.logs, "step1": step1, "summary": ""}

        if abs(step1["diff"]) == 0:
            summary = "No weighted-average PD change detected across the reporting dates."
            self._log(f'[Final Report] "{summary}"')
            result["summary"] = summary
            return result

        step2 = self._step2_customer_attribution()
        step3 = self._step3_root_cause_summary(step2["top_customer"])
        result.update(
            {
                "step2": step2,
                "step3": step3,
                "summary": step3["summary"],
                "top_customer": step2["top_customer"],
            }
        )
        return result


def build_in_memory_db(seed: int = 42) -> sqlite3.Connection:
    rng = random.Random(seed)
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE risk_data (
            customer_id TEXT NOT NULL,
            rating TEXT NOT NULL,
            pd REAL NOT NULL,
            exposure REAL NOT NULL,
            asset_class TEXT NOT NULL,
            reporting_date TEXT NOT NULL
        )
        """
    )

    regular_customers = [f"Customer {idx:02d}" for idx in range(1, 50)]
    rows: list[tuple[str, str, float, float, str, str]] = []

    for customer_id in regular_customers:
        asset_class = rng.choice(ASSET_CLASSES)
        rating_t0 = rng.choices(
            ["AAA", "AA", "A", "BBB", "BB"],
            weights=[0.08, 0.24, 0.36, 0.24, 0.08],
            k=1,
        )[0]

        rating_index_t0 = RATING_SCALE.index(rating_t0)
        rating_shift = rng.choices([-1, 0, 1], weights=[0.03, 0.94, 0.03], k=1)[0]
        rating_index_t1 = min(max(rating_index_t0 + rating_shift, 0), len(RATING_SCALE) - 1)
        rating_t1 = RATING_SCALE[rating_index_t1]

        pd_t0 = _clamp(
            RATING_TO_PD[rating_t0] + rng.uniform(-0.0015, 0.0015),
            0.0005,
            0.999,
        )
        pd_t1 = _clamp(
            RATING_TO_PD[rating_t1] + rng.uniform(-0.0015, 0.0015),
            0.0005,
            0.999,
        )

        exposure_t0 = rng.uniform(120_000.0, 650_000.0)
        exposure_t1 = exposure_t0 * rng.uniform(0.97, 1.03)
        rows.append(
            (
                customer_id,
                rating_t0,
                round(pd_t0, 6),
                round(exposure_t0, 2),
                asset_class,
                T0_DATE,
            )
        )
        rows.append(
            (
                customer_id,
                rating_t1,
                round(pd_t1, 6),
                round(exposure_t1, 2),
                asset_class,
                T1_DATE,
            )
        )

    rows.append((BIGCORP_NAME, "A", 0.01, 10_000_000.0, "private equity", T0_DATE))
    rows.append((BIGCORP_NAME, "BB", 0.05, 10_000_000.0, "private equity", T1_DATE))

    cursor.executemany(
        """
        INSERT INTO risk_data (customer_id, rating, pd, exposure, asset_class, reporting_date)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.commit()
    return connection


def load_risk_data(seed: int = 42) -> pd.DataFrame:
    connection = build_in_memory_db(seed=seed)
    try:
        df = pd.read_sql_query(
            """
            SELECT customer_id, rating, pd, exposure, asset_class, reporting_date
            FROM risk_data
            ORDER BY reporting_date, customer_id
            """,
            connection,
        )
    finally:
        connection.close()
    return df


def run_analysis(provider: str, api_key: str, model: str, seed: int = 42) -> dict[str, Any]:
    connection = build_in_memory_db(seed=seed)
    try:
        reasoner = OpenAIReasoner(api_key=api_key, model=model, provider=provider)
        agent = CreditAnalystAgent(
            connection=connection,
            reasoner=reasoner,
            schema_metadata=SCHEMA_METADATA,
            t0_date=T0_DATE,
            t1_date=T1_DATE,
        )
        return agent.run()
    finally:
        connection.close()


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.to_dict(orient="records")


def serialize_analysis_result(result: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "logs": result.get("logs", []),
        "summary": result.get("summary", ""),
    }

    step1 = result.get("step1")
    if step1:
        output["step1"] = {
            "sql": step1["sql"],
            "wapd_t0": step1["wapd_t0"],
            "wapd_t1": step1["wapd_t1"],
            "diff": step1["diff"],
            "raw_rows": _records(step1["raw_df"]),
        }

    step2 = result.get("step2")
    if step2:
        output["step2"] = {
            "sql": step2["sql"],
            "top_customer": step2["top_customer"],
            "contribution_sum": step2["contribution_sum"],
            "raw_rows": _records(step2["raw_df"]),
            "attribution_rows": _records(step2["attribution_df"]),
        }

    step3 = result.get("step3")
    if step3:
        output["step3"] = {
            "summary": step3["summary"],
            "details_sql": step3["details_sql"],
            "details_rows": _records(step3["details_df"]),
            "rating_change": step3["rating_change"],
            "pd_change": list(step3["pd_change"]),
            "exposure_change": list(step3["exposure_change"]),
        }

    return output
