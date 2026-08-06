import csv
import io
import os

import httpx


FRED_SERIES = {
    "FEDFUNDS": "Federal funds rate", "UNRATE": "Unemployment rate",
    "CPIAUCSL": "Consumer price index", "UMCSENT": "Consumer sentiment",
    "BAMLH0A0HYM2": "High-yield credit spread",
}


def fetch_fred_macro():
    observations, raw_parts = [], []
    for series, description in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        response = httpx.get(url, timeout=20, follow_redirects=True); response.raise_for_status()
        raw_parts.append(response.text)
        for row in csv.DictReader(io.StringIO(response.text)):
            raw_value = row.get(series)
            if raw_value in (None, ".", ""): continue
            observations.append({"series": series, "date": row["observation_date"],
                                 "value": float(raw_value), "entity": "US",
                                 "dimensions": {"description": description}})
    return "\n".join(raw_parts), observations, "https://fred.stlouisfed.org/"


def fetch_census_business_dynamics(start_year=2010):
    # The API now requires a key. This official, versioned national CSV stays
    # public and makes every experiment reproducible against the same release.
    url = ("https://www2.census.gov/programs-surveys/bds/tables/"
           "time-series/2023/bds2023.csv")
    response = httpx.get(url, timeout=30, follow_redirects=True); response.raise_for_status()
    observations = []
    for row in csv.DictReader(io.StringIO(response.text)):
        year = int(row["year"])
        if year < start_year:
            continue
        for field in ("job_creation", "job_destruction", "estabs_entry", "estabs_exit"):
            if row.get(field) not in (None, "", "null"):
                observations.append({"series": field.upper(), "date": str(year),
                                     "value": float(row[field]), "entity": "US",
                                     "dimensions": {"release": "BDS 2023"}})
    return response.text, observations, url


def fetch_sec_companyfacts(cik):
    cik = str(cik).zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    user_agent = os.getenv("SEC_USER_AGENT", "StartupIntelligence research contact@example.com")
    response = httpx.get(url, headers={"User-Agent": user_agent}, timeout=30); response.raise_for_status()
    payload = response.json(); observations = []
    wanted = {"Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "CashAndCashEquivalentsAtCarryingValue",
              "Assets", "Liabilities", "OperatingIncomeLoss", "NetIncomeLoss", "ResearchAndDevelopmentExpense"}
    for taxonomy, facts in payload.get("facts", {}).items():
        for concept, fact in facts.items():
            if concept not in wanted: continue
            for unit, values in fact.get("units", {}).items():
                for item in values:
                    if item.get("form") not in {"10-K", "10-Q"} or "val" not in item: continue
                    observations.append({"series": concept, "date": item.get("end", item.get("filed", "")),
                                         "value": float(item["val"]), "unit": unit,
                                         "entity": payload.get("entityName", cik),
                                         "dimensions": {"form": item.get("form"), "fy": item.get("fy"),
                                                        "fp": item.get("fp"), "taxonomy": taxonomy}})
    return response.text, observations, url


def parse_long_csv(csv_text):
    observations = []
    for row in csv.DictReader(io.StringIO(csv_text)):
        if not {"date", "series", "value"}.issubset(row):
            raise ValueError("CSV requires date, series, and value columns")
        observations.append({"date": row["date"], "series": row["series"],
                             "value": float(row["value"]) if row["value"] else None,
                             "unit": row.get("unit"), "entity": row.get("entity", ""),
                             "dimensions": {key: value for key, value in row.items()
                                            if key not in {"date", "series", "value", "unit", "entity"}}})
    return observations
