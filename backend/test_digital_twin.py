from ml.digital_twin import predict_digital_twin


def test_trained_digital_twin_returns_all_prediction_heads():
    history = [{
        "revenue": 12000 + month * 1000, "mrr": 12000 + month * 1000,
        "arr": (12000 + month * 1000) * 12, "cash_on_hand": 250000 - month * 5000,
        "customer_count": 100 + month * 8, "new_customers": 12,
        "churned_customers": 4, "employee_count": 5, "marketing_spend": 3000,
        "capital_raised": 300000, "payroll_cost": 30000,
    } for month in range(8)]
    result = predict_digital_twin(history)
    predictions = result["predictions"]
    assert result["model"]["feature_count"] == 2064
    assert predictions["future_customer_count"] >= 0
    assert 0 <= predictions["cash_exhaustion_probability"] <= 1
    assert "future_revenue" in predictions
