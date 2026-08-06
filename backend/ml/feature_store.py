"""Leakage-safe temporal feature engineering for the startup digital twin.

Raw observations stay understandable and auditable. This module expands them into
lags, rolling statistics, momentum, volatility, and unit-economics ratios suitable
for multi-target models. Features at month t never inspect a future month.
"""

from collections.abc import Mapping, Sequence

import numpy as np


RAW_SIGNAL_GROUPS = {
    "financial": (
        "revenue", "mrr", "arr", "cash_on_hand", "gross_profit", "cogs",
        "payroll_cost", "marketing_spend", "sales_spend", "rd_spend",
        "general_admin_spend", "debt", "accounts_receivable", "accounts_payable",
    ),
    "customers": (
        "customer_count", "new_customers", "churned_customers", "expanded_customers",
        "contracted_customers", "reactivated_customers", "failed_payments",
        "enterprise_customers", "smb_customers", "customer_concentration_top10",
    ),
    "acquisition": (
        "website_visitors", "signups", "qualified_leads", "sales_opportunities",
        "won_deals", "organic_leads", "paid_leads", "partner_leads",
        "ad_impressions", "ad_clicks", "sales_cycle_days",
    ),
    "product": (
        "dau", "wau", "mau", "sessions", "core_actions", "activated_users",
        "invited_users", "api_calls", "uptime_percent", "p95_latency_ms",
        "critical_incidents", "features_shipped",
    ),
    "retention_support": (
        "nps", "csat", "support_tickets", "median_first_response_hours",
        "median_resolution_hours", "open_critical_tickets", "refunds",
        "logo_retention", "gross_revenue_retention", "net_revenue_retention",
    ),
    "team_execution": (
        "employee_count", "engineering_count", "sales_count", "support_count",
        "marketing_count", "new_hires", "departures", "open_roles",
        "deployment_count", "lead_time_days", "change_failure_rate",
    ),
    "fundraising": (
        "investor_meetings", "investor_followups", "term_sheets", "capital_raised",
        "valuation", "founder_ownership_percent", "months_since_last_raise",
    ),
    "market": (
        "market_growth_index", "competitor_count", "competitor_funding",
        "competitor_price_index", "search_interest_index", "interest_rate",
        "unemployment_rate", "inflation_rate", "venture_funding_index",
        "business_formation_index",
    ),
}

RAW_SIGNALS = tuple(signal for group in RAW_SIGNAL_GROUPS.values() for signal in group)
WINDOWS = (3, 6, 12)
LAGS = (1, 2, 3, 6, 12)


def _number(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _safe_divide(numerator, denominator):
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-9:
        return np.nan
    return numerator / denominator


def _slope(values):
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.nan
    x = np.arange(len(values))[finite]
    return float(np.polyfit(x, values[finite], 1)[0])


def build_temporal_features(history: Sequence[Mapping], as_of_index: int | None = None):
    """Create a wide feature vector using observations available at ``as_of_index``."""
    if not history:
        raise ValueError("At least one monthly observation is required")
    end = len(history) - 1 if as_of_index is None else as_of_index
    if end < 0 or end >= len(history):
        raise IndexError("as_of_index is outside the supplied history")

    features = {}
    for signal in RAW_SIGNALS:
        series = np.array([_number(month.get(signal)) for month in history[: end + 1]])
        current = series[-1]
        features[f"{signal}__current"] = current
        features[f"{signal}__missing"] = float(not np.isfinite(current))

        for lag in LAGS:
            features[f"{signal}__lag_{lag}"] = series[-lag - 1] if len(series) > lag else np.nan

        previous = series[-2] if len(series) > 1 else np.nan
        features[f"{signal}__delta_1"] = current - previous
        features[f"{signal}__pct_change_1"] = _safe_divide(current - previous, abs(previous))

        for window in WINDOWS:
            values = series[-window:]
            valid = values[np.isfinite(values)]
            prefix = f"{signal}__rolling_{window}"
            features[f"{prefix}_mean"] = float(np.mean(valid)) if len(valid) else np.nan
            features[f"{prefix}_std"] = float(np.std(valid)) if len(valid) else np.nan
            features[f"{prefix}_min"] = float(np.min(valid)) if len(valid) else np.nan
            features[f"{prefix}_max"] = float(np.max(valid)) if len(valid) else np.nan
            features[f"{prefix}_slope"] = _slope(values)

    # Cross-domain unit economics and operating efficiency features.
    current = {signal: _number(history[end].get(signal)) for signal in RAW_SIGNALS}
    ratio_specs = {
        "gross_margin": ("gross_profit", "revenue"),
        "burn_multiple": ("marketing_spend", "mrr"),
        "revenue_per_employee": ("revenue", "employee_count"),
        "mrr_per_customer": ("mrr", "customer_count"),
        "cac_proxy": ("marketing_spend", "new_customers"),
        "lead_to_signup": ("signups", "qualified_leads"),
        "signup_conversion": ("new_customers", "signups"),
        "sales_win_rate": ("won_deals", "sales_opportunities"),
        "visitor_to_signup": ("signups", "website_visitors"),
        "activation_rate": ("activated_users", "signups"),
        "dau_mau": ("dau", "mau"),
        "wau_mau": ("wau", "mau"),
        "actions_per_active_user": ("core_actions", "mau"),
        "tickets_per_customer": ("support_tickets", "customer_count"),
        "revenue_concentration_risk": ("customer_concentration_top10", "revenue"),
        "engineering_share": ("engineering_count", "employee_count"),
        "sales_share": ("sales_count", "employee_count"),
        "support_share": ("support_count", "employee_count"),
        "rd_intensity": ("rd_spend", "revenue"),
        "sales_marketing_intensity": ("sales_spend", "revenue"),
        "debt_to_revenue": ("debt", "revenue"),
        "cash_to_monthly_spend": ("cash_on_hand", "payroll_cost"),
        "term_sheet_rate": ("term_sheets", "investor_meetings"),
        "funding_efficiency": ("revenue", "capital_raised"),
    }
    for name, (numerator, denominator) in ratio_specs.items():
        features[f"ratio__{name}"] = _safe_divide(current[numerator], current[denominator])

    return features


def build_supervised_rows(history: Sequence[Mapping], horizon_months=3, minimum_history=3):
    """Build features at t and labels from t+h, preventing target leakage."""
    if horizon_months < 1:
        raise ValueError("horizon_months must be positive")
    rows = []
    for as_of in range(minimum_history - 1, len(history) - horizon_months):
        future = history[as_of + horizon_months]
        current = history[as_of]
        current_revenue = _number(current.get("revenue"))
        future_revenue = _number(future.get("revenue"))
        rows.append({
            "as_of_index": as_of,
            "features": build_temporal_features(history, as_of),
            "labels": {
                "future_revenue": future_revenue,
                "future_customer_count": _number(future.get("customer_count")),
                "future_cash_on_hand": _number(future.get("cash_on_hand")),
                "revenue_growth": _safe_divide(future_revenue - current_revenue, abs(current_revenue)),
                "cash_exhausted": float(_number(future.get("cash_on_hand")) <= 0),
            },
        })
    return rows
