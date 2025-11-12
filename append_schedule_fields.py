import re
import pandas as pd
from pathlib import Path

# === File paths 
FLIGHTS_FILE = "flight_legs_with_weather_all-2.csv"   
SCHEDULE_FILE = "W24_25_schedule_Jan-Mar.xlsx"      
SCHEDULE_SHEET = 0                                 
OUTPUT_FILE = "flights_with_schedule_fields.csv"

def smart_read_flights_csv(path: str) -> pd.DataFrame:
    path = Path(path)
    try:
        df = pd.read_csv(path, engine="python", sep=None, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path, engine="python", sep=None)

    if df.shape[1] == 1:
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            header_line = f.readline()
        delim = ";" if ";" in header_line and "," not in header_line else ","
        df = pd.read_csv(path, engine="python", sep=delim, encoding="utf-8-sig")

    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    rename_map = {
        "acutal_departure_time": "actual_departure_time",
        "Origin-Latitute": "Origin-Latitude",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df

def normalize_flight_key_from_string(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().upper()
    m = re.search(r'([A-Z]{1,3})?\s*[-\s]*0*([0-9]{1,5})', s)
    if not m:
        return re.sub(r'[^A-Z0-9]', '', s)
    letters = m.group(1) or ""
    digits = m.group(2).lstrip("0") or "0"
    return f"{letters}{digits}"

def normalize_flight_key_from_cols(al, flno) -> str:
    al = "" if pd.isna(al) else str(al).strip().upper()
    if pd.isna(flno) or str(flno).strip() == "":
        num = ""
    else:
        s = str(flno).strip()
        if re.fullmatch(r"\d+", s):
            num = str(int(s))
        else:
            num = (re.sub(r"\D", "", s).lstrip("0") or "0")
    return f"{al}{num}"

def parse_flight_date_csv(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=False)  # m/d/yyyy

def parse_schedule_date_excel(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True)   # d.m.yyyy

def norm_airport(x):
    return "" if pd.isna(x) else str(x).strip().upper()

# ---------- Load data ----------
print(f"Reading flights from {FLIGHTS_FILE} ...")
flights = smart_read_flights_csv(FLIGHTS_FILE)

required_flight_cols = ["flight_number", "flight_date_yyyy_MM_dd", "origin", "destination"]
missing = [c for c in required_flight_cols if c not in flights.columns]
if missing:
    raise ValueError(f"Flights CSV is missing required columns after parsing: {missing}\n"
                     f"Got columns: {list(flights.columns)}")

print(f"Reading schedule from {SCHEDULE_FILE} (sheet={SCHEDULE_SHEET}) ...")
schedule = pd.read_excel(SCHEDULE_FILE, sheet_name=SCHEDULE_SHEET, dtype={"FlNo": str, "Al": str})

sched_cols = ["Month_L", "Weekday_L", "Date_L", "Al", "FlNo", "Orig", "Dest", "A/C"]
missing_sched = [c for c in sched_cols if c not in schedule.columns]
if missing_sched:
    raise ValueError(f"Schedule Excel is missing expected columns: {missing_sched}")

# ---------- Prepare keys & dates ----------
flights["flight_key"] = flights["flight_number"].apply(normalize_flight_key_from_string)
flights["flight_date"] = parse_flight_date_csv(flights["flight_date_yyyy_MM_dd"]).dt.date
flights["origin_norm"] = flights["origin"].map(norm_airport)
flights["destination_norm"] = flights["destination"].map(norm_airport)

schedule = schedule[sched_cols].copy()
schedule["flight_key"] = schedule.apply(lambda r: normalize_flight_key_from_cols(r["Al"], r["FlNo"]), axis=1)
schedule["flight_date"] = parse_schedule_date_excel(schedule["Date_L"]).dt.date
schedule["origin_norm"] = schedule["Orig"].map(norm_airport)
schedule["destination_norm"] = schedule["Dest"].map(norm_airport)

sched_small = schedule[["flight_key","flight_date","origin_norm","destination_norm",
                        "Month_L","Weekday_L","A/C"]].drop_duplicates()

# ---------- Merge: strict then loose ----------
# Strict: date + flight + route
strict = pd.merge(
    flights, sched_small, how="left",
    on=["flight_key","flight_date","origin_norm","destination_norm"]
)

# Loose: only date + flight, for rows still missing — avoid duplicate columns by selecting only the keys
needs_loose = strict["Month_L"].isna()
if needs_loose.any():
    left_loose = strict.loc[needs_loose, ["flight_key", "flight_date"]].copy()
    left_loose["__rowid__"] = strict.index[needs_loose]

    sched_loose = schedule[["flight_key","flight_date","Month_L","Weekday_L","A/C"]].drop_duplicates()
    loose_merged = pd.merge(left_loose, sched_loose, how="left", on=["flight_key","flight_date"])

    for col in ["Month_L","Weekday_L","A/C"]:
        strict.loc[loose_merged["__rowid__"], col] = loose_merged[col].values

# ---------- Rename ----------
strict = strict.rename(columns={
    "Month_L": "schedule_month",
    "Weekday_L": "schedule_weekday",
    "A/C": "aircraft_type"
})

matched = strict["schedule_month"].notna().sum()
total = len(strict)
print(f"\nMatched schedule for {matched}/{total} flights ({matched/total:.1%}).")

# ---------- NEW: drop rows where any of the schedule fields are missing ----------
# Convert blanks/whitespace to NaN first
strict[["schedule_month","schedule_weekday","aircraft_type"]] = (
    strict[["schedule_month","schedule_weekday","aircraft_type"]]
    .replace(r"^\s*$", pd.NA, regex=True)
)

before = len(strict)
strict = strict.dropna(subset=["schedule_month","schedule_weekday","aircraft_type"])
after = len(strict)
print(f"Filtered out {before - after} rows missing schedule fields. Kept {after} rows.")

# ---------- Save ----------
strict.to_csv(OUTPUT_FILE, index=False)
print(f" Saved -> {OUTPUT_FILE}")


