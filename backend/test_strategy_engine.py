from prediction_engine import train_churn_model, train_growth_model
from strategy_engine import analyze_strategies


def test_strategy_lab_ranks_multiple_decisions_and_is_reproducible():
    churn_model, _ = train_churn_model()
    growth_model, _ = train_growth_model()
    state = {
        "cash_on_hand": 150000,
        "customer_count": 120,
        "price_per_customer": 50,
        "marketing_spend": 3000,
        "employee_count": 3,
    }

    first = analyze_strategies(state, churn_model, growth_model, simulations=50, seed=7)
    second = analyze_strategies(state, churn_model, growth_model, simulations=50, seed=7)

    assert first == second
    assert first["strategies_evaluated"] >= 27
    assert first["best_strategy"]["rank"] == 1
    assert 0 <= first["best_strategy"]["survival_probability"] <= 1
    assert len(first["top_strategies"]) == 5
