AVERAGE_SALARY = 6000
TOOLING_ESTIMATE = 300


def compute_monthly_snapshot(previous_snapshot, customer_count, employee_count, marketing_spend):
    revenue = customer_count * previous_snapshot["price_per_customer"]
    payroll = employee_count * AVERAGE_SALARY
    monthly_costs = payroll + marketing_spend + TOOLING_ESTIMATE
    burn_rate = monthly_costs - revenue
    cash_on_hand = previous_snapshot["cash_on_hand"] - burn_rate

    if burn_rate > 0:
        runway_months = cash_on_hand / burn_rate
    else:
        runway_months = float("inf")

    growth_rate = (customer_count - previous_snapshot["customer_count"]) / previous_snapshot["customer_count"]

    return {
        "revenue": revenue,
        "monthly_costs": monthly_costs,
        "burn_rate": burn_rate,
        "cash_on_hand": cash_on_hand,
        "runway_months": runway_months,
        "growth_rate": growth_rate,
        "customer_count": customer_count,
        "employee_count": employee_count,
        "marketing_spend": marketing_spend,
    }
