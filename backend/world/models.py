from dataclasses import asdict, dataclass, field
from uuid import uuid4


@dataclass
class Company:
    id: str
    name: str
    cash: float
    customers: int
    price: float
    marketing: float
    engineers: int
    salespeople: int
    support: int
    product_quality: float = 0.55
    technical_debt: float = 0.3
    reputation: float = 0.5
    founder_ownership: float = 0.9
    revenue: float = 0
    alive: bool = True


@dataclass
class CustomerSegment:
    id: str
    name: str
    population: int
    budget: float
    price_sensitivity: float
    quality_preference: float
    switching_cost: float
    growth_rate: float


@dataclass
class InvestorMarket:
    available_capital: float = 100_000_000
    risk_appetite: float = 0.55
    valuation_multiple: float = 6.0


@dataclass
class MacroEconomy:
    regime: str = "stable"
    demand_multiplier: float = 1.0
    interest_rate: float = 0.05
    unemployment_rate: float = 0.045
    venture_sentiment: float = 0.55


@dataclass
class WorldState:
    id: str
    name: str
    seed: int
    month: int = 0
    branch_id: str = "main"
    parent_branch_id: str | None = None
    companies: dict[str, Company] = field(default_factory=dict)
    segments: dict[str, CustomerSegment] = field(default_factory=dict)
    investors: InvestorMarket = field(default_factory=InvestorMarket)
    macro: MacroEconomy = field(default_factory=MacroEconomy)

    def to_dict(self):
        return asdict(self)


def create_world(name="Startup Civilization", seed=2026):
    focal = Company("player", "Player Startup", 500_000, 180, 75, 8_000, 3, 1, 1)
    competitors = {
        "competitor_alpha": Company("competitor_alpha", "Atlas Labs", 900_000, 260, 89, 14_000, 5, 2, 1, 0.62, 0.25, 0.58),
        "competitor_beta": Company("competitor_beta", "Nova Systems", 350_000, 120, 49, 6_000, 2, 2, 1, 0.48, 0.4, 0.42),
    }
    segments = {
        "smb": CustomerSegment("smb", "Small businesses", 12_000, 85, 0.8, 0.45, 0.18, 0.018),
        "midmarket": CustomerSegment("midmarket", "Mid-market", 3_500, 240, 0.45, 0.7, 0.42, 0.012),
        "enterprise": CustomerSegment("enterprise", "Enterprise", 700, 900, 0.2, 0.9, 0.72, 0.007),
    }
    return WorldState(
        id=str(uuid4()), name=name, seed=seed,
        companies={focal.id: focal, **competitors}, segments=segments,
    )


def world_from_dict(data):
    return WorldState(
        id=data["id"], name=data["name"], seed=data["seed"], month=data.get("month", 0),
        branch_id=data.get("branch_id", "main"), parent_branch_id=data.get("parent_branch_id"),
        companies={key: Company(**value) for key, value in data["companies"].items()},
        segments={key: CustomerSegment(**value) for key, value in data["segments"].items()},
        investors=InvestorMarket(**data["investors"]), macro=MacroEconomy(**data["macro"]),
    )
