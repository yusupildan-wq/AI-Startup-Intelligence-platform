"""Coherent synthetic startup populations for pipeline development.

This is explicitly a simulator, not real-world evidence. Latent company traits create
correlated finance, product, team, sales, and market histories so the full digital-
twin training path can be developed before licensed/user data is available.
"""

import numpy as np

from ml.feature_store import RAW_SIGNALS


def generate_startup_history(months=24, seed=0):
    rng = np.random.default_rng(seed)
    market_fit = rng.beta(2.2, 2.0)
    product_quality = rng.beta(3.0, 1.8)
    execution = rng.beta(3.0, 2.0)
    sales_efficiency = rng.beta(2.3, 2.2)
    capital_access = rng.beta(2.0, 2.5)
    price = rng.uniform(25, 220)
    customers = int(rng.integers(15, 400))
    employees = int(rng.integers(2, 18))
    cash = rng.uniform(80_000, 2_000_000)
    capital_raised = cash
    history = []

    for month in range(1, months + 1):
        cycle = np.sin((month + seed % 12) / 12 * 2 * np.pi)
        market = np.clip(0.9 + 0.18 * cycle + rng.normal(0, 0.08), 0.55, 1.35)
        marketing = max(500, cash * rng.uniform(0.004, 0.018))
        visitors = max(50, marketing * (1.2 + 2.2 * market_fit) * market / rng.uniform(1.5, 4.0))
        signups = rng.binomial(round(visitors), np.clip(0.025 + 0.12 * product_quality, 0, 0.5))
        activated = rng.binomial(signups, np.clip(0.25 + 0.6 * product_quality, 0, 0.95))
        qualified = rng.binomial(signups, np.clip(0.2 + 0.55 * market_fit, 0, 0.9))
        opportunities = rng.binomial(qualified, np.clip(0.2 + 0.5 * sales_efficiency, 0, 0.9))
        won = rng.binomial(opportunities, np.clip(0.12 + 0.5 * sales_efficiency, 0, 0.85))
        organic = rng.poisson(max(1, customers * market_fit * product_quality * 0.035))
        new_customers = int(won + organic)
        churn_probability = np.clip(0.16 - 0.1 * product_quality - 0.035 * market_fit + price / 9000, 0.008, 0.3)
        churned = rng.binomial(customers, churn_probability)
        customers = max(0, customers + new_customers - churned)

        employee_growth = 1 if cash > 250_000 and rng.random() < 0.12 * execution else 0
        departures = rng.binomial(employees, np.clip(0.025 + (0.04 if cash < 60_000 else 0), 0, 0.2))
        employees = max(1, employees + employee_growth - departures)
        engineering = max(1, round(employees * (0.35 + 0.2 * product_quality)))
        sales = max(0, round(employees * (0.12 + 0.2 * sales_efficiency)))
        support = max(0, round(employees * 0.12))
        marketing_count = max(0, round(employees * 0.1))

        mrr = customers * price
        revenue = mrr * rng.uniform(0.96, 1.05)
        cogs = revenue * np.clip(0.32 - 0.18 * execution, 0.06, 0.5)
        gross_profit = revenue - cogs
        payroll = employees * rng.uniform(5_000, 9_000)
        sales_spend = sales * rng.uniform(1200, 3500)
        rd_spend = engineering * rng.uniform(900, 2800)
        ga_spend = employees * rng.uniform(250, 650)
        burn = payroll + marketing + sales_spend + rd_spend + ga_spend + cogs - revenue
        cash -= burn

        investor_meetings = int(rng.poisson(2 + 8 * capital_access)) if cash < 12 * max(burn, 1) else 0
        term_sheets = rng.binomial(investor_meetings, 0.03 + 0.22 * capital_access) if investor_meetings else 0
        raised = float(term_sheets * rng.uniform(150_000, 1_500_000))
        cash += raised
        capital_raised += raised

        mau = max(customers, round(customers * rng.uniform(2, 8)))
        wau = round(mau * (0.25 + 0.45 * product_quality))
        dau = round(wau * (0.18 + 0.45 * product_quality))
        tickets = rng.poisson(max(1, customers * (0.08 - 0.05 * product_quality)))
        row = {signal: 0.0 for signal in RAW_SIGNALS}
        row.update({
            "revenue": revenue, "mrr": mrr, "arr": mrr * 12, "cash_on_hand": cash,
            "gross_profit": gross_profit, "cogs": cogs, "payroll_cost": payroll,
            "marketing_spend": marketing, "sales_spend": sales_spend, "rd_spend": rd_spend,
            "general_admin_spend": ga_spend, "debt": max(0, -cash * 0.25),
            "accounts_receivable": revenue * rng.uniform(0.05, 0.3),
            "accounts_payable": (cogs + ga_spend) * rng.uniform(0.1, 0.6),
            "customer_count": customers, "new_customers": new_customers,
            "churned_customers": churned, "expanded_customers": rng.binomial(customers, 0.03),
            "contracted_customers": rng.binomial(customers, 0.015),
            "reactivated_customers": rng.poisson(max(0.2, churned * 0.05)),
            "failed_payments": rng.binomial(customers, 0.012),
            "enterprise_customers": round(customers * 0.08), "smb_customers": round(customers * 0.72),
            "customer_concentration_top10": min(revenue, revenue * rng.uniform(0.12, 0.7)),
            "website_visitors": visitors, "signups": signups, "qualified_leads": qualified,
            "sales_opportunities": opportunities, "won_deals": won,
            "organic_leads": round(qualified * 0.5), "paid_leads": round(qualified * 0.35),
            "partner_leads": round(qualified * 0.15), "ad_impressions": visitors * rng.uniform(8, 25),
            "ad_clicks": visitors, "sales_cycle_days": 70 - 40 * sales_efficiency,
            "dau": dau, "wau": wau, "mau": mau, "sessions": dau * rng.uniform(1.3, 3.5),
            "core_actions": mau * rng.uniform(2, 15) * product_quality, "activated_users": activated,
            "invited_users": activated * rng.uniform(0.1, 1.2), "api_calls": mau * rng.uniform(5, 500),
            "uptime_percent": 98.5 + 1.45 * product_quality, "p95_latency_ms": 700 - 500 * product_quality,
            "critical_incidents": rng.poisson(1.5 * (1 - product_quality)),
            "features_shipped": rng.poisson(1 + 4 * execution), "nps": -10 + 80 * product_quality,
            "csat": 2.2 + 2.6 * product_quality, "support_tickets": tickets,
            "median_first_response_hours": 30 / (1 + support), "median_resolution_hours": 70 / (1 + support),
            "open_critical_tickets": rng.binomial(tickets, 0.04) if tickets else 0, "refunds": churned * price * 0.1,
            "logo_retention": 1 - churn_probability, "gross_revenue_retention": 1 - churn_probability * 0.85,
            "net_revenue_retention": 1 - churn_probability * 0.85 + 0.03 * market_fit,
            "employee_count": employees, "engineering_count": engineering, "sales_count": sales,
            "support_count": support, "marketing_count": marketing_count, "new_hires": employee_growth,
            "departures": departures, "open_roles": int(cash > 300_000),
            "deployment_count": rng.poisson(2 + engineering * execution), "lead_time_days": 35 / (1 + execution),
            "change_failure_rate": 0.25 * (1 - product_quality), "investor_meetings": investor_meetings,
            "investor_followups": round(investor_meetings * capital_access), "term_sheets": term_sheets,
            "capital_raised": capital_raised, "valuation": max(revenue * 12 * rng.uniform(3, 10), capital_raised),
            "founder_ownership_percent": max(15, 100 - 8 * (capital_raised > 0) - month * 0.1),
            "months_since_last_raise": 0 if raised else month, "market_growth_index": market,
            "competitor_count": rng.integers(4, 45), "competitor_funding": rng.uniform(1e6, 80e6),
            "competitor_price_index": rng.uniform(0.7, 1.4), "search_interest_index": 50 * market,
            "interest_rate": 5 - 1.8 * cycle, "unemployment_rate": 4.5 - cycle,
            "inflation_rate": 2.5 + 0.8 * cycle, "venture_funding_index": 100 * market,
            "business_formation_index": 100 * market,
        })
        history.append(row)
    return history


def generate_population(companies=160, months=24, seed=2026):
    return [generate_startup_history(months, seed + company * 997) for company in range(companies)]
