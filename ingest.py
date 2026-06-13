from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st


REQUIRED_COLUMNS = [
    "Player Name",
    "Team",
    "Grade",
    "Position",
    "Strengths",
    "Development Areas",
    "Projection",
    "Event Name",
    "Event Date",
    "Overall Score",
]

OPTIONAL_COLUMNS = ["Growth Upside"]
NUMERIC_COLUMNS = ["Overall Score", "Growth Upside"]

DEFAULT_DATA_FILES = [
    Path("scout_chatbot/scouting_database.csv"),
    Path("scout_chatbot/AAU_Scouting_System.xlsx"),
]


def _extract_event_start_date(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()
    if not text:
        return pd.NaT

    return pd.to_datetime(text.split(" - ", maxsplit=1)[0].strip(), errors="coerce")


def _normalize_workbook(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    evaluations = sheets["Player_Evaluations"].copy()
    events = sheets["Event_Log"].copy()

    evaluations.columns = [str(column).strip() for column in evaluations.columns]
    events.columns = [str(column).strip() for column in events.columns]

    merged = evaluations.merge(events[["Event ID", "Event Name", "Date"]], on="Event ID", how="left")
    return pd.DataFrame(
        {
            "Player Name": merged["Player Name"],
            "Team": merged["Team"],
            "Grade": merged["Level"],
            "Position": merged["Position"],
            "Strengths": merged["Strengths"],
            "Development Areas": merged["Development Areas"],
            "Projection": merged["Projection"],
            "Event Name": merged["Event Name"],
            "Event Date": merged["Date"].map(_extract_event_start_date),
            "Overall Score": merged["Overall Grade"],
            "Growth Upside": merged["Growth Upside (1-5)"],
        }
    )


def _read_dataframe(source) -> pd.DataFrame:
    source_name = getattr(source, "name", str(source))
    suffix = Path(source_name).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(source)

    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(source, sheet_name=None)
        normalized_sheets = {str(sheet_name).strip(): dataframe for sheet_name, dataframe in sheets.items()}

        if "Player_Evaluations" in normalized_sheets and "Event_Log" in normalized_sheets:
            return _normalize_workbook(normalized_sheets)

        for dataframe in normalized_sheets.values():
            candidate = dataframe.copy()
            candidate.columns = [str(column).strip() for column in candidate.columns]
            if all(column in candidate.columns for column in REQUIRED_COLUMNS):
                return candidate

    raise ValueError("Unsupported file type. Upload a CSV or Excel scouting export.")


def prepare_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_text}")

    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    selected_columns = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    df = df[selected_columns].copy()
    df["Event Date"] = pd.to_datetime(df["Event Date"], errors="coerce")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in [
        "Player Name",
        "Team",
        "Grade",
        "Position",
        "Strengths",
        "Development Areas",
        "Projection",
        "Event Name",
    ]:
        df[column] = df[column].fillna("Unknown").astype(str).str.strip()
        df.loc[df[column] == "", column] = "Unknown"

    df = df.dropna(subset=["Overall Score"])
    df = df.sort_values(["Player Name", "Event Date", "Event Name"]).reset_index(drop=True)
    return df


def build_document_text(record: dict[str, object]) -> str:
    growth_upside = record.get("Growth Upside")
    upside_text = (
        f"Growth Upside: {float(growth_upside):.2f}. "
        if pd.notna(growth_upside)
        else "Growth Upside: Not provided. "
    )

    event_date = record.get("Event Date")
    if isinstance(event_date, pd.Timestamp) and not pd.isna(event_date):
        event_date_text = event_date.strftime("%Y-%m-%d")
    else:
        event_date_text = str(event_date)

    return (
        f"Player Name: {record['Player Name']}. "
        f"Team: {record['Team']}. "
        f"Grade: {record['Grade']}. "
        f"Position: {record['Position']}. "
        f"Strengths: {record['Strengths']}. "
        f"Development Areas: {record['Development Areas']}. "
        f"Projection: {record['Projection']}. "
        f"Event Name: {record['Event Name']}. "
        f"Event Date: {event_date_text}. "
        f"Overall Score: {float(record['Overall Score']):.2f}. "
        f"{upside_text}"
    )


def build_records(df: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        record = dict(row)
        record["document"] = build_document_text(record)
        records.append(record)
    return records


def dataframe_signature(df: pd.DataFrame) -> str:
    serialized = df.to_json(orient="records", date_format="iso")
    return sha256(serialized.encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False)
def load_data_from_path(path: str) -> pd.DataFrame:
    return prepare_dataframe(_read_dataframe(path))


def load_data_from_upload(uploaded_file) -> pd.DataFrame:
    return prepare_dataframe(_read_dataframe(uploaded_file))


def find_default_data_file() -> Path | None:
    for path in DEFAULT_DATA_FILES:
        if path.exists():
            return path
    return None


def load_chatbot_data() -> tuple[pd.DataFrame, str]:
    uploaded_file = st.sidebar.file_uploader(
        "Upload scouting export",
        type=["csv", "xlsx", "xls"],
        help="Upload a scouting database export from Google Sheets. Excel files are also supported.",
    )

    if uploaded_file is not None:
        return load_data_from_upload(uploaded_file), uploaded_file.name

    default_file = find_default_data_file()
    if default_file is not None:
        return load_data_from_path(str(default_file)), default_file.name

    raise FileNotFoundError("No data file found. Upload a scouting export to continue.")


def apply_filters(
    df: pd.DataFrame,
    teams: list[str] | None = None,
    grades: list[str] | None = None,
    positions: list[str] | None = None,
    events: list[str] | None = None,
) -> pd.DataFrame:
    filtered_df = df.copy()
    filters = {
        "Team": teams or [],
        "Grade": grades or [],
        "Position": positions or [],
        "Event Name": events or [],
    }

    for column, values in filters.items():
        if values:
            filtered_df = filtered_df[filtered_df[column].isin(values)]

    return filtered_df.reset_index(drop=True)
