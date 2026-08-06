"""Deterministic event engine with replayable and branchable timelines."""

from copy import deepcopy
from uuid import uuid4

import numpy as np

from ml.population_models import (
    customer_purchase_probability, employee_departure_probability,
    product_adoption_probability,
)
from ml.economy_agents import competitor_action, investor_offer, macro_regime
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
        if not shock:
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
        company.last_action = action
        company.last_funding_raised = 0
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
            monthly_cost = (company.engineers + company.salespeople + company.support) * 7_000 + company.marketing + 1_500
            burn = monthly_cost - company.revenue
            runway = company.cash / max(burn, 1) if burn > 0 else 36
            probability, proposed = investor_offer([
                0, company.revenue, company.cash, min(36, runway), company.product_quality,
                company.reputation, self.state.macro.venture_sentiment,
                self.state.macro.interest_rate, company.founder_ownership,
            ])
            company.last_funding_probability = probability
            if probability >= .5:
                raised = min(self.state.investors.available_capital, proposed)
                dilution = min(0.25, raised / max(raised + company.cash + company.revenue * 48, 1))
                company.cash += raised; company.founder_ownership *= 1 - dilution
                company.last_funding_raised = raised
                self.state.investors.available_capital -= raised

    def _resolve_company_month(self, company, rng, month):
        macro = self.state.macro
        competitors = [item for item in self.state.companies.values() if item.id != company.id and item.alive]
        average_quality = np.mean([item.product_quality for item in competitors]) if competitors else company.product_quality
        average_price = np.mean([item.price for item in competitors]) if competitors else company.price
        relative_quality = company.product_quality - average_quality
        relative_price = company.price / max(average_price, 1)
        addressable = sum(segment.population * (1 + segment.growth_rate) ** month for segment in self.state.segments.values())
        competitor_utility = float(np.mean([
            item.product_quality + item.reputation - item.price / 300 for item in competitors
        ])) if competitors else 0.0
        acquired = 0
        adoption_rates = []
        for segment in self.state.segments.values():
            purchase_probability = customer_purchase_probability(
                segment, company, competitor_utility, macro.demand_multiplier
            )
            reachable = min(round(segment.population * 0.015),
                            round(np.sqrt(company.marketing) * (0.35 + company.salespeople * 0.08)))
            acquired += int(rng.binomial(max(0, reachable), np.clip(purchase_probability, 0, 1)))
            adoption_rates.append(product_adoption_probability(
                segment.quality_preference, company.product_quality, 1 - company.technical_debt,
                company.reputation, segment.switching_cost, relative_price - 1,
                min(1, company.customers / max(segment.population, 1)),
            ))
        churn_rate = np.clip(0.11 - company.product_quality * 0.065 - company.support * 0.004 + max(0, relative_price - 1) * 0.04, 0.006, 0.25)
        churned = int(rng.binomial(company.customers, churn_rate))
        customers = min(round(addressable), max(0, company.customers + acquired - churned))
        revenue = customers * company.price
        previous_revenue = company.revenue
        revenue_growth = (revenue - previous_revenue) / max(previous_revenue, 1) if previous_revenue else 0
        current_headcount = company.engineers + company.salespeople + company.support
        monthly_burn = current_headcount * 7_000 + company.marketing + 1_500 - revenue
        runway = company.cash / max(monthly_burn, 1) if monthly_burn > 0 else 36
        departure_probability = employee_departure_probability(
            1.0, np.clip((company.reputation + company.product_quality) / 2, 0, 1),
            company.technical_debt, month, revenue_growth, min(36, runway),
            np.clip(1 - company.technical_debt, 0, 1), np.clip(1 - macro.unemployment_rate * 5, 0, 1),
        )
        engineer_departures = int(rng.binomial(company.engineers, departure_probability))
        sales_departures = int(rng.binomial(company.salespeople, departure_probability))
        support_departures = int(rng.binomial(company.support, departure_probability))
        engineers = max(0, company.engineers - engineer_departures)
        salespeople = max(0, company.salespeople - sales_departures)
        support = max(0, company.support - support_departures)
        payroll = (engineers + salespeople + support) * 7_000
        costs = payroll + company.marketing + 1_500 + revenue * 0.12
        cash = company.cash + revenue - costs
        adoption_rate = float(np.mean(adoption_rates))
        quality = np.clip(company.product_quality + 0.004 * np.log1p(engineers) - 0.01 * company.technical_debt, 0.05, 1)
        debt = np.clip(company.technical_debt + 0.012 - 0.003 * engineers, 0, 1)
        reputation = np.clip(company.reputation + (acquired - churned) / max(customers, 1) * 0.025 + (adoption_rate - .5) * .01, 0.05, 1)
        return WorldEvent(month, "company_month_resolved", company.id, {
            "cash": float(cash), "customers": customers, "revenue": float(revenue),
            "product_quality": float(quality), "technical_debt": float(debt),
            "reputation": float(reputation), "alive": bool(cash > 0 and customers > 0),
            "engineers": engineers, "salespeople": salespeople, "support": support,
            "customers_acquired": acquired, "customers_churned": churned,
            "employees_departed": engineer_departures + sales_departures + support_departures,
            "product_adoption_rate": adoption_rate,
        })

    def _competitor_action(self, company, rng):
        rivals = [item for item in self.state.companies.values() if item.id != company.id and item.alive]
        average_price = np.mean([item.price for item in rivals]) if rivals else company.price
        average_quality = np.mean([item.product_quality for item in rivals]) if rivals else company.product_quality
        monthly_cost = (company.engineers + company.salespeople + company.support) * 7_000 + company.marketing + 1_500
        return competitor_action([
            company.cash / max(monthly_cost, 1), 0, company.price / max(average_price, 1),
            company.product_quality - average_quality, company.marketing / max(company.revenue, 1),
            company.technical_debt, self.state.macro.demand_multiplier,
            self.state.macro.venture_sentiment,
            min(1, (company.engineers + company.salespeople + company.support) / 20),
            min(1, len(rivals) / 10),
        ])

    def _advance_macro(self, rng, month):
        macro = self.state.macro
        cycle = np.sin(month / 12 * 2 * np.pi)
        predicted = macro_regime([
            macro.demand_multiplier, macro.interest_rate, macro.unemployment_rate,
            macro.venture_sentiment, .025 + .012 * max(0, cycle),
            macro.venture_sentiment - .5, cycle * .12, cycle * .18,
        ])
        targets = {
            "recession": (.72, .18, .08, .085), "stable": (1.0, .52, .05, .05),
            "expansion": (1.2, .7, .035, .038), "funding_boom": (1.28, .92, .025, .035),
        }
        demand_target, sentiment_target, rate_target, unemployment_target = targets[predicted]
        return WorldEvent(month, "macro_updated", "world", {
            "regime": predicted,
            "demand_multiplier": float(np.clip(.75 * macro.demand_multiplier + .25 * demand_target, .55, 1.45)),
            "venture_sentiment": float(np.clip(.75 * macro.venture_sentiment + .25 * sentiment_target, 0, 1)),
            "interest_rate": float(.8 * macro.interest_rate + .2 * rate_target),
            "unemployment_rate": float(.8 * macro.unemployment_rate + .2 * unemployment_target),
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
