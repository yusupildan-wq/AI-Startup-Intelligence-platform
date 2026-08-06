from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class WorldEvent:
    month: int
    type: str
    actor_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self):
        return asdict(self)


ACTION_TYPES = {
    "hold", "raise_price", "lower_price", "increase_marketing", "decrease_marketing",
    "hire_engineer", "hire_sales", "hire_support", "reduce_headcount", "fundraise",
    "invest_in_product", "enter_new_market",
}

SHOCK_TYPES = {"recession", "funding_boom", "demand_surge", "technology_shift"}
