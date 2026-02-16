import random
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st
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
GROUPABLE_COLUMNS = ["reporting_date", "asset_class", "rating", "customer_id"]
AGGREGATION_OPTIONS = {
    "row_count": "Row Count",
    "sum_exposure": "Sum Exposure",
    "avg_exposure": "Average Exposure",
    "avg_pd": "Average PD",
    "weighted_avg_pd": "Weighted Average PD",
}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:sql)?\\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\\s*```$", "", cleaned)
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
        if re.search(rf"\\b{token}\\b", lower_sql):
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


def _discover_available_models(api_key: str, provider: str) -> list[str]:
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
        print(message)

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

        rating_changed = old_rating != new_rating
        rating_note = "No rating change"
        if rating_changed:
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
            "details_sql_template": "\n".join(line.strip() for line in details_sql.strip().splitlines()),
            "details_params": details_params,
            "details_df": details_df,
            "rating_change": rating_note,
            "pd_change": (old_pd, new_pd),
            "exposure_change": (old_exposure, new_exposure),
        }

    def run(self) -> dict[str, Any]:
        step1 = self._step1_portfolio_trigger()
        result: dict[str, Any] = {
            "logs": self.logs,
            "step1": step1,
            "summary": "",
        }

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
                "attribution_df": step2["attribution_df"],
                "details_df": step3["details_df"],
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


@st.cache_data(show_spinner=False)
def _load_risk_data(seed: int = 42) -> pd.DataFrame:
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


def _render_sidebar_data_controls(
    base_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    st.sidebar.header("Data Controls")

    reporting_date_options = sorted(base_df["reporting_date"].unique().tolist())
    rating_options = sorted(base_df["rating"].unique().tolist(), key=RATING_SCALE.index)
    asset_class_options = [item for item in ASSET_CLASSES if item in set(base_df["asset_class"])]

    selected_reporting_dates = st.sidebar.multiselect(
        "Comparison Dates (choose 2)",
        options=reporting_date_options,
        default=reporting_date_options,
        max_selections=2,
    )
    selected_asset_classes = st.sidebar.multiselect(
        "Filter: Asset Class",
        options=asset_class_options,
        default=asset_class_options,
    )
    selected_ratings = st.sidebar.multiselect(
        "Filter: Rating",
        options=rating_options,
        default=rating_options,
    )
    customer_search = st.sidebar.text_input("Filter: Customer contains", value="")

    pd_min, pd_max = float(base_df["pd"].min()), float(base_df["pd"].max())
    selected_pd_range = st.sidebar.slider(
        "Filter: PD range",
        min_value=pd_min,
        max_value=pd_max,
        value=(pd_min, pd_max),
    )

    exposure_min, exposure_max = float(base_df["exposure"].min()), float(base_df["exposure"].max())
    selected_exposure_range = st.sidebar.slider(
        "Filter: Exposure range (EUR)",
        min_value=exposure_min,
        max_value=exposure_max,
        value=(exposure_min, exposure_max),
        step=10_000.0,
    )

    group_by_columns = st.sidebar.multiselect(
        "Group By",
        options=GROUPABLE_COLUMNS,
        default=["reporting_date"],
    )
    selected_aggregations = st.sidebar.multiselect(
        "Aggregations",
        options=list(AGGREGATION_OPTIONS.keys()),
        format_func=lambda item: AGGREGATION_OPTIONS[item],
        default=["row_count", "sum_exposure", "weighted_avg_pd"],
    )

    filtered_df = base_df.copy()
    filtered_df = filtered_df[filtered_df["reporting_date"].isin(selected_reporting_dates)]
    filtered_df = filtered_df[filtered_df["asset_class"].isin(selected_asset_classes)]
    filtered_df = filtered_df[filtered_df["rating"].isin(selected_ratings)]
    filtered_df = filtered_df[
        (filtered_df["pd"] >= selected_pd_range[0]) & (filtered_df["pd"] <= selected_pd_range[1])
    ]
    filtered_df = filtered_df[
        (filtered_df["exposure"] >= selected_exposure_range[0])
        & (filtered_df["exposure"] <= selected_exposure_range[1])
    ]

    if customer_search.strip():
        filtered_df = filtered_df[
            filtered_df["customer_id"].str.contains(customer_search.strip(), case=False, na=False)
        ]

    filtered_df = filtered_df.sort_values(["reporting_date", "customer_id"]).reset_index(drop=True)
    return filtered_df, group_by_columns, selected_aggregations, selected_reporting_dates


def _build_aggregated_view(
    filtered_df: pd.DataFrame,
    group_by_columns: list[str],
    selected_aggregations: list[str],
) -> pd.DataFrame | None:
    if not selected_aggregations:
        return None

    working_df = filtered_df.copy()
    working_df["_pd_x_exposure"] = working_df["pd"] * working_df["exposure"]

    def _weighted_average_pd(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return float("nan")
        return numerator / denominator

    if group_by_columns:
        grouped = working_df.groupby(group_by_columns, dropna=False)
        metric_series: list[pd.Series] = []

        if "row_count" in selected_aggregations:
            metric_series.append(grouped.size().rename("row_count"))
        if "sum_exposure" in selected_aggregations:
            metric_series.append(grouped["exposure"].sum().rename("sum_exposure"))
        if "avg_exposure" in selected_aggregations:
            metric_series.append(grouped["exposure"].mean().rename("avg_exposure"))
        if "avg_pd" in selected_aggregations:
            metric_series.append(grouped["pd"].mean().rename("avg_pd"))
        if "weighted_avg_pd" in selected_aggregations:
            weighted_numerator = grouped["_pd_x_exposure"].sum()
            weighted_denominator = grouped["exposure"].sum()
            weighted_avg = (weighted_numerator / weighted_denominator).rename("weighted_avg_pd")
            metric_series.append(weighted_avg)

        if not metric_series:
            return None

        return pd.concat(metric_series, axis=1).reset_index()

    single_row_metrics: dict[str, float | int] = {}
    if "row_count" in selected_aggregations:
        single_row_metrics["row_count"] = int(len(working_df))
    if "sum_exposure" in selected_aggregations:
        single_row_metrics["sum_exposure"] = float(working_df["exposure"].sum())
    if "avg_exposure" in selected_aggregations:
        single_row_metrics["avg_exposure"] = float(working_df["exposure"].mean()) if len(working_df) else float("nan")
    if "avg_pd" in selected_aggregations:
        single_row_metrics["avg_pd"] = float(working_df["pd"].mean()) if len(working_df) else float("nan")
    if "weighted_avg_pd" in selected_aggregations:
        single_row_metrics["weighted_avg_pd"] = _weighted_average_pd(
            numerator=float(working_df["_pd_x_exposure"].sum()),
            denominator=float(working_df["exposure"].sum()),
        )

    return pd.DataFrame([single_row_metrics]) if single_row_metrics else None


def _build_asset_class_comparison_view(
    filtered_df: pd.DataFrame,
    selected_reporting_dates: list[str],
) -> pd.DataFrame | None:
    if len(selected_reporting_dates) != 2:
        return None

    date_a, date_b = selected_reporting_dates[0], selected_reporting_dates[1]
    comparison_df = filtered_df[filtered_df["reporting_date"].isin([date_a, date_b])].copy()
    if comparison_df.empty:
        return None

    grouped = (
        comparison_df.groupby(["asset_class", "reporting_date"], dropna=False)
        .agg(
            exposure=("exposure", "sum"),
            avg_pd=("pd", "mean"),
        )
        .reset_index()
    )

    pivot = grouped.pivot(
        index="asset_class",
        columns="reporting_date",
        values=["exposure", "avg_pd"],
    )

    all_classes_in_scope = [item for item in ASSET_CLASSES if item in set(filtered_df["asset_class"])]
    pivot = pivot.reindex(all_classes_in_scope)

    flat_columns: list[str] = []
    for metric, report_date in pivot.columns:
        flat_columns.append(f"{metric}_{report_date}")
    pivot.columns = flat_columns
    pivot = pivot.reset_index()

    for report_date in [date_a, date_b]:
        exposure_col = f"exposure_{report_date}"
        avg_pd_col = f"avg_pd_{report_date}"
        if exposure_col not in pivot.columns:
            pivot[exposure_col] = 0.0
        if avg_pd_col not in pivot.columns:
            pivot[avg_pd_col] = float("nan")

    ordered_columns = [
        "asset_class",
        f"exposure_{date_a}",
        f"avg_pd_{date_a}",
        f"exposure_{date_b}",
        f"avg_pd_{date_b}",
    ]
    pivot = pivot[ordered_columns]
    return pivot


def _render_data_explorer_section(
    filtered_df: pd.DataFrame,
    aggregated_df: pd.DataFrame | None,
    group_by_columns: list[str],
    asset_class_comparison_df: pd.DataFrame | None,
    selected_reporting_dates: list[str],
) -> None:
    st.subheader("Portfolio Data Table")
    row_count_col, customer_count_col, exposure_sum_col = st.columns(3)
    row_count_col.metric("Rows", f"{len(filtered_df):,}")
    customer_count_col.metric("Customers", f"{filtered_df['customer_id'].nunique():,}")
    exposure_sum_col.metric("Total Exposure", f"€{filtered_df['exposure'].sum():,.2f}")

    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    st.subheader("Grouped and Aggregated View")
    if asset_class_comparison_df is None:
        st.warning("Select exactly two reporting dates to view asset-class comparison columns.")
    else:
        date_a, date_b = selected_reporting_dates[0], selected_reporting_dates[1]
        st.caption(
            f"Rows: asset classes. Columns: exposure and average PD for {date_a} and {date_b}."
        )
        display_df = asset_class_comparison_df.copy()
        for report_date in [date_a, date_b]:
            exposure_col = f"exposure_{report_date}"
            avg_pd_col = f"avg_pd_{report_date}"
            display_df[exposure_col] = display_df[exposure_col].map(lambda value: f"€{value:,.2f}")
            display_df[avg_pd_col] = display_df[avg_pd_col].map(
                lambda value: "-" if pd.isna(value) else f"{value:.4%}"
            )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    if aggregated_df is not None:
        st.subheader("Custom Aggregation View")
        if group_by_columns:
            st.caption(f"Grouped by: {', '.join(group_by_columns)}")
        else:
            st.caption("No group by selected; showing portfolio-level aggregates.")
        st.dataframe(aggregated_df, use_container_width=True, hide_index=True)


def _render_step_by_step_analysis(result: dict[str, Any], base_df: pd.DataFrame) -> None:
    st.subheader("Step-by-Step Analysis")
    st.success(f'Final note: "{result["summary"]}"')

    with st.expander("Step 1: Portfolio Level Trigger", expanded=True):
        step1 = result["step1"]
        st.caption("Generated SQL")
        st.code(step1["sql"], language="sql")
        wapd_t0_col, wapd_t1_col, diff_col = st.columns(3)
        wapd_t0_col.metric("WAPD T0", f"{step1['wapd_t0']:.4%}")
        wapd_t1_col.metric("WAPD T1", f"{step1['wapd_t1']:.4%}")
        diff_col.metric("Diff", f"{step1['diff']:+.4%}")
        st.dataframe(step1["raw_df"], use_container_width=True, hide_index=True)

    if "step2" in result:
        with st.expander("Step 2: Drill-Down Attribution", expanded=True):
            step2 = result["step2"]
            st.caption("Generated SQL")
            st.code(step2["sql"], language="sql")

            top_customer = step2["top_customer"]
            st.info(f"Primary driver identified: **{top_customer}**")

            attribution_view = step2["attribution_df"].copy()
            for column in ["pd_t0", "pd_t1", "contribution"]:
                attribution_view[column] = attribution_view[column].map(lambda value: f"{value:+.4%}")
            for column in ["exposure_t0", "exposure_t1"]:
                attribution_view[column] = attribution_view[column].map(lambda value: f"{value:,.2f}")

            st.dataframe(
                attribution_view[
                    ["customer_id", "pd_t0", "pd_t1", "exposure_t0", "exposure_t1", "contribution"]
                ],
                use_container_width=True,
                hide_index=True,
            )

            selected_customer = st.selectbox(
                "Drill-down customer",
                options=step2["attribution_df"]["customer_id"].tolist(),
                index=0,
                key="drilldown_customer",
            )
            drilldown_df = base_df[base_df["customer_id"] == selected_customer].copy()
            st.caption("Selected customer history")
            st.dataframe(drilldown_df, use_container_width=True, hide_index=True)

    if "step3" in result:
        with st.expander("Step 3: Root Cause and Summary", expanded=True):
            step3 = result["step3"]
            st.dataframe(step3["details_df"], use_container_width=True, hide_index=True)
            st.write(f"Detected change: **{step3['rating_change']}**")
            old_pd, new_pd = step3["pd_change"]
            old_exposure, new_exposure = step3["exposure_change"]
            st.write(f"PD change: `{old_pd:.4f} -> {new_pd:.4f}`")
            st.write(f"Exposure change: `€{old_exposure:,.2f} -> €{new_exposure:,.2f}`")
            st.write(f'Final report: "{step3["summary"]}"')

    if "step2" in result and "step3" in result:
        st.subheader("Drill-Down Output")
        top_customer = result["step2"]["top_customer"]
        st.write(f"Top mover: **{top_customer}**")
        st.dataframe(result["step3"]["details_df"], use_container_width=True, hide_index=True)
        st.info(f'Final note: "{result["step3"]["summary"]}"')

    st.subheader("Thought Process Log")
    st.code("\n".join(result["logs"]), language="text")


def _render_execution_window(result: dict[str, Any] | None) -> None:
    st.subheader("Execution Window")
    if result is None:
        st.info("Run the analysis to print logs, SQL queries, execution, and results.")
        return

    logs_tab, sql_tab, results_tab = st.tabs(["Log Output", "Compiled SQL", "Execution Results"])

    with logs_tab:
        st.caption("Session log output")
        st.code("\n".join(result.get("logs", [])), language="text")

    with sql_tab:
        step1 = result.get("step1", {})
        step2 = result.get("step2", {})
        step3 = result.get("step3", {})

        st.caption("Step 1 SQL")
        st.code(step1.get("sql", "--"), language="sql")

        if step2:
            st.caption("Step 2 SQL")
            st.code(step2.get("sql", "--"), language="sql")

        if step3:
            st.caption("Step 3 SQL (compiled with parameters)")
            st.code(step3.get("details_sql", "--"), language="sql")

    with results_tab:
        step1 = result.get("step1", {})
        step2 = result.get("step2", {})
        step3 = result.get("step3", {})

        if "raw_df" in step1:
            st.caption("Step 1 query results")
            st.dataframe(step1["raw_df"], use_container_width=True, hide_index=True)

        if step2 and "raw_df" in step2:
            st.caption("Step 2 query results")
            st.dataframe(step2["raw_df"], use_container_width=True, hide_index=True)

        if step2 and "attribution_df" in step2:
            st.caption("Step 2 contribution output")
            st.dataframe(step2["attribution_df"], use_container_width=True, hide_index=True)

        if step3 and "details_df" in step3:
            st.caption("Step 3 drill-down details")
            st.dataframe(step3["details_df"], use_container_width=True, hide_index=True)
            st.success(f'Final note: "{step3.get("summary", "")}"')


def render_streamlit_app() -> None:
    st.set_page_config(page_title="Credit Portfolio AI Analyst", layout="centered")
    st.title("Credit Portfolio AI Analyst")
    st.caption(
        "Iterative analysis of weighted average PD movements between two reporting dates."
    )

    base_df = _load_risk_data(seed=42)
    (
        filtered_df,
        group_by_columns,
        selected_aggregations,
        selected_reporting_dates,
    ) = _render_sidebar_data_controls(base_df)
    aggregated_df = _build_aggregated_view(filtered_df, group_by_columns, selected_aggregations)
    asset_class_comparison_df = _build_asset_class_comparison_view(
        filtered_df=filtered_df,
        selected_reporting_dates=selected_reporting_dates,
    )
    _render_data_explorer_section(
        filtered_df=filtered_df,
        aggregated_df=aggregated_df,
        group_by_columns=group_by_columns,
        asset_class_comparison_df=asset_class_comparison_df,
        selected_reporting_dates=selected_reporting_dates,
    )

    st.divider()
    st.subheader("LLM Analysis Controls")

    provider = st.selectbox("LLM Provider", options=[OPENAI_PROVIDER, GEMINI_PROVIDER], index=0)
    api_key = st.text_input(f"{provider} API Key", type="password")
    api_key_clean = api_key.strip()

    if provider == GEMINI_PROVIDER:
        st.caption(
            "Gemini uses Google's OpenAI-compatible endpoint. "
            f"Recommended model: `{DEFAULT_GEMINI_MODEL}`."
        )

    can_run = False
    available_models: list[str] = []
    selected_model = ""

    if not api_key_clean:
        st.info(f"Enter a {provider} API key to run step-by-step LLM analysis.")
    else:
        model_cache_key = f"{provider}:{api_key_clean}"
        key_changed = st.session_state.get("models_api_key") != model_cache_key
        if key_changed or "available_models" not in st.session_state:
            with st.spinner("Checking available models for this API key..."):
                try:
                    st.session_state["available_models"] = _discover_available_models(
                        api_key=api_key_clean,
                        provider=provider,
                    )
                    st.session_state["models_api_key"] = model_cache_key
                except Exception as exc:
                    st.error("Could not load available models for this API key.")
                    st.exception(exc)

        available_models = st.session_state.get("available_models", [])
        if available_models:
            preferred_default = (
                DEFAULT_OPENAI_MODEL if provider == OPENAI_PROVIDER else DEFAULT_GEMINI_MODEL
            )
            default_model = (
                preferred_default if preferred_default in available_models else available_models[0]
            )
            selected_model = st.selectbox(
                "Available Models",
                options=available_models,
                index=available_models.index(default_model),
            )
            can_run = True
        else:
            st.warning("No available models found for this API key.")

    run_clicked = st.button("Run Analysis", type="primary", disabled=not can_run)
    if run_clicked and can_run:
        connection = build_in_memory_db(seed=42)
        try:
            reasoner = OpenAIReasoner(
                api_key=api_key_clean,
                model=selected_model,
                provider=provider,
            )
            agent = CreditAnalystAgent(
                connection=connection,
                reasoner=reasoner,
                schema_metadata=SCHEMA_METADATA,
                t0_date=T0_DATE,
                t1_date=T1_DATE,
            )
            st.session_state["analysis_result"] = agent.run()
            st.session_state["analysis_provider"] = provider
            st.session_state["analysis_model"] = selected_model
        except Exception as exc:
            st.error("Analysis failed. See details below.")
            st.exception(exc)
        finally:
            connection.close()

    _render_execution_window(st.session_state.get("analysis_result"))

    if "analysis_result" in st.session_state:
        st.divider()
        st.caption(
            "Latest analysis run with "
            f"{st.session_state.get('analysis_provider', OPENAI_PROVIDER)} / "
            f"{st.session_state.get('analysis_model', DEFAULT_OPENAI_MODEL)}"
        )
        _render_step_by_step_analysis(st.session_state["analysis_result"], base_df)
    else:
        st.info("Run the analysis to view the full step-by-step drill-down output.")


if __name__ == "__main__":
    render_streamlit_app()
