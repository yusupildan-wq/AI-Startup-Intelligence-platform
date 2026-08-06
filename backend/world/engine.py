"""Deterministic event engine with replayable and branchable timelines."""

from copy import deepcopy
from uuid import uuid4

import numpy as np

from world.events import ACTION_TYPES, SHOCK_TYPES, WorldEvent
from world.models import Company, MacroEconomy, WorldState


class WorldEngine:
    def __init__(self, initial_state: WorldState):
        self.initial_state = deepcopy(initial_state)
        self.state = deepcopy(initial_state)
        self.events: list[WorldEvent] = []
        self.snapshots = {0: deepcopy(initial_state)}

    def advance(self, player_action="hold", shock=None):
        if player_action not in ACTION_TYPES:
            raise ValueError(f"Unknown action: {player_action}")
        if shock is not None and shock not in SHOCK_TYPES:
            raise ValueError(f"Unknown shock: {shock}")
        month = self.state.month + 1
        rng = np.random.default_rng(self.state.seed + month * 10_007)
        generated = []
        if shock:
            generated.append(self._shock_event(month, shock))
        generated.append(WorldEvent(month, "company_action", "player", {"action": player_action}))
        for company_id in sorted(self.state.companies):
            if company_id != "player" and self.state.companies[company_id].alive:
                action = self._competitor_action(self.state.companies[company_id], rng)
                generated.append(WorldEvent(month, "company_action", company_id, {"action": action}))

        for event in generated:
            self.apply(event)
        self.apply(self._advance_macro(rng, month))
        for company_id in sorted(self.state.companies):
            company = self.state.companies[company_id]
            if company.alive:
                outcome = self._resolve_company_month(company, rng, month)
                self.apply(outcome)
        self.state.month = month
        self.snapshots[month] = deepcopy(self.state)
        return deepcopy(self.state), [event for event in self.events if event.month == month]

    def apply(self, event: WorldEvent):
        if event.type == "company_action":
            self._apply_action(self.state.companies[event.actor_id], event.payload["action"])
        elif event.type == "company_month_resolved":
            company = self.state.companies[event.actor_id]
            for key, value in event.payload.items():
                setattr(company, key, value)
        elif event.type in {"macro_shock", "macro_updated"}:
            for key, value in event.payload.items():
                setattr(self.state.macro, key, value)
        self.events.append(event)

    def branch(self, at_month: int, branch_name=None):
        if at_month not in self.snapshots:
            raise ValueError("Cannot branch from a month without a snapshot")
        branched_state = deepcopy(self.snapshots[at_month])
        branched_state.parent_branch_id = self.state.branch_id
        branched_state.branch_id = branch_name or f"branch-{str(uuid4())[:8]}"
        engine = WorldEngine(branched_state)
        engine.initial_state = deepcopy(branched_state)
        return engine

    def replay(self):
        replayed = WorldEngine(self.initial_state)
        for month in sorted({event.month for event in self.events}):
            for event in [item for item in self.events if item.month == month]:
                replayed.apply(event)
            replayed.state.month = month
            replayed.snapshots[month] = deepcopy(replayed.state)
        return replayed.state

    def _apply_action(self, company: Company, action: str):
        if action == "raise_price": company.price *= 1.12
        elif action == "lower_price": company.price *= 0.9
        elif action == "increase_marketing": company.marketing = company.marketing * 1.4 + 500
        elif action == "decrease_marketing": company.marketing = max(250, company.marketing * 0.7)
        elif action == "hire_engineer": company.engineers += 1
        elif action == "hire_sales": company.salespeople += 1
        elif action == "hire_support": company.support += 1
        elif action == "reduce_headcount":
            role = max(("engineers", "salespeople", "support"), key=lambda item: getattr(company, item))
            setattr(company, role, max(0, getattr(company, role) - 1))
        elif action == "invest_in_product":
            company.cash -= 25_000; company.product_quality = min(1, company.product_quality + 0.05); company.technical_debt = max(0, company.technical_debt - 0.06)
        elif action == "enter_new_market": company.cash -= 40_000; company.reputation = min(1, company.reputation + 0.03)
        elif action == "fundraise":
            raised = min(self.state.investors.available_capital, max(0, company.revenue * 18 + 250_000))
            dilution = min(0.25, raised / max(raised + company.cash + company.revenue * 48, 1))
            company.cash += raised; company.founder_ownership *= 1 - dilution
            self.state.investors.available_capital -= raised

    def _resolve_company_month(self, company, rng, month):
        macro = self.state.macro
        competitors = [item for item in self.state.companies.values() if item.id != company.id and item.alive]
        relative_quality = company.product_quality - np.mean([item.product_quality for item in competitors])
        relative_price = company.price / max(np.mean([item.price for item in competitors]), 1)
        addressable = sum(segment.population * (1 + segment.growth_rate) ** month for segment in self.state.segments.values())
        appeal = np.clip(0.18 + 0.5 * company.product_quality + 0.22 * company.reputation + 0.08 * relative_quality - 0.16 * max(0, relative_price - 1), 0.02, 0.95)
        acquired = int(rng.poisson(max(1, np.sqrt(company.marketing) * appeal * macro.demand_multiplier + company.salespeople * 4)))
        churn_rate = np.clip(0.11 - company.product_quality * 0.065 - company.support * 0.004 + max(0, relative_price - 1) * 0.04, 0.006, 0.25)
        churned = int(rng.binomial(company.customers, churn_rate))
        customers = min(round(addressable), max(0, company.customers + acquired - churned))
        revenue = customers * company.price
        payroll = (company.engineers + company.salespeople + company.support) * 7_000
        costs = payroll + company.marketing + 1_500 + revenue * 0.12
        cash = company.cash + revenue - costs
        quality = np.clip(company.product_quality + 0.004 * np.log1p(company.engineers) - 0.01 * company.technical_debt, 0.05, 1)
        debt = np.clip(company.technical_debt + 0.012 - 0.003 * company.engineers, 0, 1)
        reputation = np.clip(company.reputation + (acquired - churned) / max(customers, 1) * 0.025, 0.05, 1)
        return WorldEvent(month, "company_month_resolved", company.id, {
            "cash": float(cash), "customers": customers, "revenue": float(revenue),
            "product_quality": float(quality), "technical_debt": float(debt),
            "reputation": float(reputation), "alive": bool(cash > 0 and customers > 0),
        })

    def _competitor_action(self, company, rng):
        if company.cash < 100_000: return "decrease_marketing" if company.marketing > 1000 else "fundraise"
        choices = ["hold", "raise_price", "increase_marketing", "hire_engineer", "hire_sales", "invest_in_product"]
        return str(rng.choice(choices))

    def _advance_macro(self, rng, month):
        macro = self.state.macro
        return WorldEvent(month, "macro_updated", "world", {
            "demand_multiplier": float(np.clip(0.88 * macro.demand_multiplier + 0.12 * rng.normal(1, 0.12), 0.55, 1.45)),
            "venture_sentiment": float(np.clip(0.85 * macro.venture_sentiment + 0.15 * rng.uniform(0.25, 0.9), 0, 1)),
        })

    def _shock_event(self, month, shock):
        macro = self.state.macro
        payloads = {
            "recession": {"regime": "recession", "demand_multiplier": 0.68, "venture_sentiment": 0.2, "interest_rate": 0.08},
            "funding_boom": {"regime": "funding_boom", "venture_sentiment": 0.92, "interest_rate": 0.025},
            "demand_surge": {"regime": "demand_surge", "demand_multiplier": 1.38},
            "technology_shift": {"regime": "technology_shift", "demand_multiplier": 1.1},
        }
        return WorldEvent(month, "macro_shock", "world", payloads[shock])
