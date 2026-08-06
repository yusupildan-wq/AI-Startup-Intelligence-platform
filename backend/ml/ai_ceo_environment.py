"""A controllable startup environment for reinforcement-learning experiments."""

from dataclasses import dataclass

import numpy as np


ACTIONS = (
    "hold", "raise_price", "lower_price", "increase_marketing", "decrease_marketing",
    "hire_engineer", "hire_sales", "hire_support", "reduce_headcount",
    "fundraise", "invest_in_product", "enter_new_market",
)


@dataclass
class StartupState:
    cash: float
    customers: float
    price: float
    marketing: float
    engineers: float
    salespeople: float
    support: float
    product_quality: float
    market_fit: float
    tech_debt: float
    ownership: float
    market: float
    month: int = 0

    def vector(self):
        return np.array([
            self.cash / 1_000_000, self.customers / 2_000, self.price / 200,
            self.marketing / 50_000, self.engineers / 20, self.salespeople / 20,
            self.support / 20, self.product_quality, self.market_fit, self.tech_debt,
            self.ownership, self.market, self.month / 36,
        ], dtype=np.float32)


class StartupEnvironment:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.state = None

    def reset(self):
        self.state = StartupState(
            cash=float(self.rng.uniform(100_000, 1_500_000)),
            customers=float(self.rng.integers(30, 600)), price=float(self.rng.uniform(30, 160)),
            marketing=float(self.rng.uniform(1_000, 25_000)), engineers=float(self.rng.integers(1, 8)),
            salespeople=float(self.rng.integers(0, 6)), support=float(self.rng.integers(0, 4)),
            product_quality=float(self.rng.uniform(0.3, 0.85)), market_fit=float(self.rng.uniform(0.25, 0.85)),
            tech_debt=float(self.rng.uniform(0.1, 0.65)), ownership=float(self.rng.uniform(0.65, 1.0)),
            market=float(self.rng.uniform(0.75, 1.25)), month=0,
        )
        return self.state.vector()

    def set_state(self, state):
        self.state = StartupState(**state.__dict__)
        return self.state.vector()

    def step(self, action_index):
        s = self.state
        action = ACTIONS[action_index]
        previous_value = self._company_value()

        if action == "raise_price": s.price *= 1.12
        elif action == "lower_price": s.price *= 0.9
        elif action == "increase_marketing": s.marketing = min(s.marketing * 1.4 + 500, 80_000)
        elif action == "decrease_marketing": s.marketing = max(250, s.marketing * 0.7)
        elif action == "hire_engineer": s.engineers += 1
        elif action == "hire_sales": s.salespeople += 1
        elif action == "hire_support": s.support += 1
        elif action == "reduce_headcount":
            largest = max(("engineers", "salespeople", "support"), key=lambda role: getattr(s, role))
            setattr(s, largest, max(0, getattr(s, largest) - 1))
        elif action == "invest_in_product":
            s.cash -= 25_000; s.product_quality = min(1, s.product_quality + 0.045); s.tech_debt = max(0, s.tech_debt - 0.06)
        elif action == "enter_new_market":
            s.cash -= 40_000; s.market_fit = np.clip(s.market_fit + self.rng.normal(0.025, 0.06), 0.05, 1)
        elif action == "fundraise":
            probability = np.clip(0.08 + 0.38 * s.market_fit + 0.2 * s.market - 0.18 * s.tech_debt, 0.05, 0.8)
            if self.rng.random() < probability:
                raised = self.rng.uniform(350_000, 1_500_000)
                dilution = min(0.28, raised / max(raised + self._company_value(), 1))
                s.cash += raised; s.ownership *= 1 - dilution
            else:
                s.cash -= 12_000

        engineer_effect = np.log1p(s.engineers) * (1 - 0.55 * s.tech_debt)
        sales_effect = np.log1p(s.salespeople) * s.market_fit
        support_effect = np.log1p(s.support) * 0.018
        price_resistance = max(0, s.price / 180 - s.product_quality * 0.45)
        acquisition_mean = (
            np.sqrt(s.marketing) * 0.35 * s.market_fit * s.market
            + 6 * sales_effect + s.customers * 0.012 * s.product_quality
        )
        acquired = self.rng.poisson(max(0.1, acquisition_mean))
        churn_rate = np.clip(0.13 - 0.075 * s.product_quality - support_effect + 0.055 * price_resistance, 0.008, 0.28)
        churned = self.rng.binomial(round(max(0, s.customers)), churn_rate)
        s.customers = max(0, s.customers + acquired - churned)
        payroll = (s.engineers + s.salespeople + s.support) * 7_000
        revenue = s.customers * s.price
        costs = payroll + s.marketing + 1_500 + revenue * 0.12
        s.cash += revenue - costs
        s.product_quality = np.clip(s.product_quality + 0.004 * engineer_effect - 0.012 * s.tech_debt, 0.05, 1)
        s.tech_debt = np.clip(s.tech_debt + 0.012 - 0.004 * s.engineers, 0, 1)
        s.market = float(np.clip(0.86 * s.market + 0.14 * self.rng.normal(1, 0.16), 0.55, 1.45))
        s.month += 1

        done = s.cash <= 0 or s.customers <= 0 or s.month >= 36
        new_value = self._company_value()
        reward = (new_value - previous_value) / 100_000
        reward += 0.08 if s.cash > 0 else -12
        reward -= max(0, 0.55 - s.ownership) * 0.4
        if s.month == 36 and s.cash > 0: reward += 4
        return s.vector(), float(reward), done, {
            "action": action, "revenue": revenue, "cash": s.cash, "customers": round(s.customers),
            "ownership": s.ownership, "company_value": new_value,
        }

    def _company_value(self):
        s = self.state
        revenue = s.customers * s.price
        quality_multiplier = 0.6 + s.product_quality + 0.5 * s.market_fit
        return max(0, revenue * 12 * quality_multiplier + max(0, s.cash) * 0.6) * s.ownership
