import math

import numpy as np

from ml.feature_store import RAW_SIGNALS, build_supervised_rows, build_temporal_features


def make_history(months=18):
    history = []
    for month in range(1, months + 1):
        row = {signal: month * (index + 1) for index, signal in enumerate(RAW_SIGNALS)}
        row.update({"month_number": month, "revenue": month * 1000, "mrr": month * 100,
                    "cash_on_hand": 100000 - month * 1000, "employee_count": 5,
                    "customer_count": month * 10, "new_customers": month,
                    "marketing_spend": 2000, "gross_profit": month * 700})
        history.append(row)
    return history


def test_feature_store_creates_hundreds_of_auditable_features():
    features = build_temporal_features(make_history())
    assert len(RAW_SIGNALS) >= 75
    assert len(features) >= 1500
    assert features["revenue__current"] == 18000
    assert features["revenue__lag_3"] == 15000
    assert features["ratio__gross_margin"] == 0.7


def test_features_never_look_past_as_of_month():
    history = make_history()
    before = build_temporal_features(history, as_of_index=8)
    history[17]["revenue"] = 999999999
    after = build_temporal_features(history, as_of_index=8)
    assert before.keys() == after.keys()
    assert np.allclose(list(before.values()), list(after.values()), equal_nan=True)


def test_supervised_rows_use_future_only_for_labels():
    rows = build_supervised_rows(make_history(10), horizon_months=3)
    first = rows[0]
    assert first["as_of_index"] == 2
    assert first["features"]["revenue__current"] == 3000
    assert first["labels"]["future_revenue"] == 6000
    assert math.isclose(first["labels"]["revenue_growth"], 1.0)
