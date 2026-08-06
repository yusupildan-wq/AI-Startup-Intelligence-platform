import pytest

from data import connectors
from data.connectors import parse_long_csv


class FakeResponse:
    text = ("year,job_creation,job_destruction,estabs_entry,estabs_exit\n"
            "2019,90,70,18,9\n2022,100,80,20,10\n")

    def raise_for_status(self):
        return None


def test_census_connector_filters_and_normalizes(monkeypatch):
    monkeypatch.setattr(connectors.httpx, "get", lambda *args, **kwargs: FakeResponse())
    _, observations, url = connectors.fetch_census_business_dynamics(2020)
    assert len(observations) == 4
    assert observations[0]["series"] == "JOB_CREATION"
    assert observations[0]["date"] == "2022"
    assert "bds2023.csv" in url


def test_long_csv_preserves_lineage_dimensions():
    rows = parse_long_csv("date,series,value,unit,entity,channel\n2026-01,MRR,12000,USD,Acme,organic\n")
    assert rows == [{"date": "2026-01", "series": "MRR", "value": 12000.0,
                     "unit": "USD", "entity": "Acme", "dimensions": {"channel": "organic"}}]


def test_long_csv_rejects_missing_required_columns():
    with pytest.raises(ValueError):
        parse_long_csv("month,revenue\n1,100\n")
