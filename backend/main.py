import secrets

import bcrypt
import openai
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from state_store import insert_startup, get_startup, get_all_snapshots, insert_user, get_user_by_email
from orchestrator import run_month
from auth import create_token, verify_token
from prediction_engine import benchmark_churn_models, train_growth_model, train_fundraising_model
from prediction_engine import train_churn_model
from strategy_engine import analyze_strategies
from state_store import get_latest_snapshot
from ml.digital_twin import load_digital_twin, predict_digital_twin
from ml.ai_ceo import load_ai_ceo, recommend_action, state_from_startup
from ml.population_models import load_population_models
from ml.economy_agents import load_economy_agents
from ml.world_generator import generate_learned_world, load_world_generator
from ml.trajectory_model import generate_trajectories, load_trajectory_model
from ml.registry import model_registry
from ml.causal_actions import estimate_action_effects
from ml.model_based_ceo import plan_actions
from world import WorldEngine, create_world
from world.events import ACTION_TYPES, SHOCK_TYPES
from world.store import (
    assert_world_owner, create_branch_record, create_world_record, ensure_world_tables,
    list_branches, list_events, list_snapshots as list_world_snapshots, list_worlds, load_engine, persist_advance,
)
from data.connectors import fetch_census_business_dynamics, fetch_fred_macro, fetch_sec_companyfacts, parse_long_csv
from data.store import dataset_observations, ensure_data_tables, list_datasets, save_dataset

app = FastAPI()

_churn_model_benchmark = benchmark_churn_models()
_, _growth_model_metrics = train_growth_model()
_, _fundraising_model_metrics = train_fundraising_model()
_strategy_churn_model, _ = train_churn_model()
_strategy_growth_model, _ = train_growth_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class CreateStartupRequest(BaseModel):
    name: str
    business_type: str
    initial_price: float
    founder_count: int
    initial_funding: float
    initial_customer_count: int = 0


class SimulateMonthRequest(BaseModel):
    marketing_spend: float
    employee_count: int
    attempt_fundraising: bool = False


class StrategyLabRequest(BaseModel):
    horizon_months: int = 12
    simulations: int = 250


class CreateWorldRequest(BaseModel):
    name: str = "Startup Civilization"
    seed: int = 2026
    startup_id: int | None = None
    generator: str = "learned"
    scenario: str = "balanced"


class AdvanceWorldRequest(BaseModel):
    action: str = "hold"
    shock: str | None = None


class BranchWorldRequest(BaseModel):
    from_month: int
    name: str


class GenerateTrajectoriesRequest(BaseModel):
    action: str = "hold"
    horizon: int = 12
    paths: int = 150
    seed: int = 2028


class ModelBasedPlanRequest(BaseModel):
    horizon: int = 12
    beam_width: int = 10
    paths: int = 60
    risk_aversion: float = 0.65
    seed: int = 932


class HumanAiComparisonRequest(BaseModel):
    human_action: str
    risk_aversion: float = 0.65
    seed: int = 1776


class OfficialDataImportRequest(BaseModel):
    source: str
    cik: str | None = None
    start_year: int = 2010


class CsvDataImportRequest(BaseModel):
    dataset_name: str
    csv_text: str


_world_storage_ready = False
_data_storage_ready = False


def ensure_world_storage():
    global _world_storage_ready
    if not _world_storage_ready:
        ensure_world_tables()
        _world_storage_ready = True


def ensure_data_storage():
    global _data_storage_ready
    if not _data_storage_ready:
        ensure_data_tables()
        _data_storage_ready = True


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/model-metrics")
def model_metrics():
    return {
        "churn_model_comparison": _churn_model_benchmark,
        "growth_model": _growth_model_metrics,
        "fundraising_model": _fundraising_model_metrics,
        "digital_twin": load_digital_twin()["metrics"],
        "ai_ceo": load_ai_ceo()["metrics"],
        "population_models": load_population_models()["metrics"],
        "economy_agents": load_economy_agents()["metrics"],
        "world_generator": load_world_generator()["metrics"],
        "trajectory_model": load_trajectory_model()["metrics"],
    }


@app.post("/register")
def register(request: RegisterRequest):
    if get_user_by_email(request.email) is not None:
        raise HTTPException(status_code=400, detail="Email already registered")

    password_hash = bcrypt.hashpw(request.password.encode(), bcrypt.gensalt())
    user_id = insert_user(request.email, password_hash.decode())
    return {"user_id": user_id}


@app.post("/login")
def login(request: LoginRequest):
    user = get_user_by_email(request.email)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not bcrypt.checkpw(request.password.encode(), user["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(user["id"])
    return {"access_token": token}


@app.post("/guest-login")
def guest_login():
    guest_email = f"guest_{secrets.token_hex(8)}@guest.local"
    guest_password = secrets.token_hex(16)

    password_hash = bcrypt.hashpw(guest_password.encode(), bcrypt.gensalt())
    user_id = insert_user(guest_email, password_hash.decode())

    token = create_token(user_id)
    return {"access_token": token}


@app.post("/startups")
def create_startup(request: CreateStartupRequest, user_id: int = Depends(verify_token)):
    startup_id = insert_startup(
        request.name,
        request.business_type,
        request.initial_price,
        request.founder_count,
        request.initial_funding,
        request.initial_customer_count,
        user_id=user_id,
    )
    return {"startup_id": startup_id}


def get_owned_startup_or_403(startup_id, user_id):
    startup = get_startup(startup_id)
    if startup is None:
        raise HTTPException(status_code=404, detail="Startup not found")
    if startup["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not your startup")
    return startup


@app.get("/startups/{startup_id}/snapshots")
def list_snapshots(startup_id: int, user_id: int = Depends(verify_token)):
    get_owned_startup_or_403(startup_id, user_id)
    return get_all_snapshots(startup_id)


@app.get("/startups/{startup_id}/digital-twin")
def digital_twin_forecast(startup_id: int, user_id: int = Depends(verify_token)):
    startup = get_owned_startup_or_403(startup_id, user_id)
    snapshots = get_all_snapshots(startup_id)
    if not snapshots:
        snapshots = [{
            "revenue": startup["initial_customer_count"] * float(startup["initial_price"]),
            "cash_on_hand": startup["initial_funding"],
            "customer_count": startup["initial_customer_count"],
            "customers_acquired": 0, "customers_churned": 0,
            "employee_count": max(1, startup["founder_count"]),
            "marketing_spend": 0, "funding_raised_to_date": startup["initial_funding"],
        }]
    history = []
    for snapshot in snapshots:
        revenue = float(snapshot["revenue"])
        customers = int(snapshot["customer_count"])
        employees = int(snapshot["employee_count"])
        history.append({
            "revenue": revenue, "mrr": revenue, "arr": revenue * 12,
            "cash_on_hand": float(snapshot["cash_on_hand"]), "customer_count": customers,
            "new_customers": int(snapshot.get("customers_acquired", 0)),
            "churned_customers": int(snapshot.get("customers_churned", 0)),
            "employee_count": employees, "marketing_spend": float(snapshot["marketing_spend"]),
            "capital_raised": float(snapshot.get("funding_raised_to_date", 0)),
            "payroll_cost": employees * 6000,
        })
    result = predict_digital_twin(history)
    result["observed_months"] = len(history)
    result["data_coverage"] = {
        "observed_signals": 11,
        "total_signals": 85,
        "coverage_percent": round(11 / 85 * 100, 1),
        "warning": "Early synthetic model with limited live inputs; not financial advice.",
    }
    return result


@app.get("/startups/{startup_id}/ai-ceo")
def ai_ceo_decision(startup_id: int, user_id: int = Depends(verify_token)):
    startup = get_owned_startup_or_403(startup_id, user_id)
    latest = get_latest_snapshot(startup_id)
    state = state_from_startup(startup, latest)
    result = recommend_action(state)
    result["current_state"] = {
        "month": state.month, "cash": round(state.cash, 2), "customers": round(state.customers),
        "price": round(state.price, 2), "marketing": round(state.marketing, 2),
        "employees": round(state.engineers + state.salespeople + state.support),
    }
    return result


@app.post("/startups/{startup_id}/ai-ceo/execute")
def execute_ai_ceo_decision(startup_id: int, user_id: int = Depends(verify_token)):
    startup = get_owned_startup_or_403(startup_id, user_id)
    latest = get_latest_snapshot(startup_id)
    decision = recommend_action(state_from_startup(startup, latest), rollout_months=1)["recommendation"]
    marketing_spend = float(latest["marketing_spend"]) if latest else 1000.0
    employee_count = int(latest["employee_count"]) if latest else max(1, startup["founder_count"])
    simulation = run_month(
        startup_id=startup_id,
        marketing_spend=marketing_spend,
        employee_count=employee_count,
        ai_action=decision["action"],
    )
    return {"executed_decision": decision, "simulation": simulation}


@app.post("/startups/{startup_id}/simulate-next-month")
def simulate_next_month(startup_id: int, request: SimulateMonthRequest, user_id: int = Depends(verify_token)):
    get_owned_startup_or_403(startup_id, user_id)
    try:
        return run_month(
            startup_id=startup_id,
            marketing_spend=request.marketing_spend,
            employee_count=request.employee_count,
            attempt_fundraising=request.attempt_fundraising,
        )
    except openai.OpenAIError as exc:
        raise HTTPException(
            status_code=503,
            detail="OpenAI narration is temporarily unavailable. The simulation was not completed.",
        ) from exc


@app.post("/startups/{startup_id}/strategy-lab")
def strategy_lab(startup_id: int, request: StrategyLabRequest, user_id: int = Depends(verify_token)):
    startup = get_owned_startup_or_403(startup_id, user_id)
    if not 3 <= request.horizon_months <= 36:
        raise HTTPException(status_code=422, detail="Horizon must be between 3 and 36 months")
    if not 50 <= request.simulations <= 2000:
        raise HTTPException(status_code=422, detail="Simulations must be between 50 and 2000")

    latest = get_latest_snapshot(startup_id)
    starting_state = latest or {
        "cash_on_hand": startup["initial_funding"],
        "customer_count": startup["initial_customer_count"],
        "price_per_customer": startup["initial_price"],
        "marketing_spend": 1000,
        "employee_count": max(1, startup["founder_count"]),
    }
    return analyze_strategies(
        starting_state,
        _strategy_churn_model,
        _strategy_growth_model,
        horizon_months=request.horizon_months,
        simulations=request.simulations,
    )


@app.get("/worlds")
def get_worlds(user_id: int = Depends(verify_token)):
    ensure_world_storage()
    return list_worlds(user_id)


@app.post("/worlds")
def create_simulation_world(request: CreateWorldRequest, user_id: int = Depends(verify_token)):
    ensure_world_storage()
    if request.generator not in {"learned", "template"}:
        raise HTTPException(status_code=422, detail="Generator must be learned or template")
    if request.scenario not in {"balanced", "recession", "funding_boom", "technology_shift"}:
        raise HTTPException(status_code=422, detail="Unknown generation scenario")
    world = (generate_learned_world(request.name, request.seed, request.scenario)
             if request.generator == "learned" else create_world(request.name, request.seed))
    if request.startup_id is not None:
        startup = get_owned_startup_or_403(request.startup_id, user_id)
        latest = get_latest_snapshot(request.startup_id)
        player = world.companies["player"]
        player.name = startup["name"]
        player.cash = float(latest["cash_on_hand"] if latest else startup["initial_funding"])
        player.customers = int(latest["customer_count"] if latest else startup["initial_customer_count"])
        player.price = float(latest["price_per_customer"] if latest else startup["initial_price"])
        if latest:
            player.marketing = float(latest["marketing_spend"])
            employees = int(latest["employee_count"])
            player.engineers = max(1, round(employees * 0.5))
            player.salespeople = max(0, round(employees * 0.25))
            player.support = max(0, employees - player.engineers - player.salespeople)
    engine = WorldEngine(world)
    create_world_record(user_id, engine)
    return engine.state.to_dict()


def get_owned_world_engine(world_id, branch_id, user_id):
    ensure_world_storage()
    if not assert_world_owner(world_id, user_id):
        raise HTTPException(status_code=404, detail="World not found")
    engine = load_engine(world_id, branch_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Branch not found")
    return engine


@app.get("/worlds/{world_id}/branches")
def get_world_branches(world_id: str, user_id: int = Depends(verify_token)):
    ensure_world_storage()
    if not assert_world_owner(world_id, user_id):
        raise HTTPException(status_code=404, detail="World not found")
    return list_branches(world_id)


@app.get("/worlds/{world_id}/branches/{branch_id}")
def inspect_world(world_id: str, branch_id: str, user_id: int = Depends(verify_token)):
    return get_owned_world_engine(world_id, branch_id, user_id).state.to_dict()


@app.get("/worlds/{world_id}/branches/{branch_id}/replay/{month}")
def replay_world_month(world_id: str, branch_id: str, month: int,
                       user_id: int = Depends(verify_token)):
    engine = get_owned_world_engine(world_id, branch_id, user_id)
    if month not in engine.snapshots:
        raise HTTPException(status_code=404, detail="Snapshot not found for this month")
    return engine.snapshots[month].to_dict()


@app.post("/worlds/{world_id}/branches/{branch_id}/advance")
def advance_world(world_id: str, branch_id: str, request: AdvanceWorldRequest,
                  user_id: int = Depends(verify_token)):
    if request.action not in ACTION_TYPES:
        raise HTTPException(status_code=422, detail=f"Action must be one of: {sorted(ACTION_TYPES)}")
    if request.shock is not None and request.shock not in SHOCK_TYPES:
        raise HTTPException(status_code=422, detail=f"Shock must be one of: {sorted(SHOCK_TYPES)}")
    engine = get_owned_world_engine(world_id, branch_id, user_id)
    state, events = engine.advance(request.action, request.shock)
    persist_advance(engine, events)
    return {"state": state.to_dict(), "events": [event.to_dict() for event in events]}


@app.post("/worlds/{world_id}/branches/{branch_id}/branch")
def branch_world(world_id: str, branch_id: str, request: BranchWorldRequest,
                 user_id: int = Depends(verify_token)):
    engine = get_owned_world_engine(world_id, branch_id, user_id)
    try:
        branch = engine.branch(request.from_month, request.name)
        create_branch_record(user_id, engine, branch, request.name, request.from_month)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return branch.state.to_dict()


@app.get("/worlds/{world_id}/branches/{branch_id}/events")
def get_world_events(world_id: str, branch_id: str, user_id: int = Depends(verify_token)):
    get_owned_world_engine(world_id, branch_id, user_id)
    return list_events(world_id, branch_id)


@app.get("/worlds/{world_id}/compare-branches")
def compare_world_branches(world_id: str, left_branch: str, right_branch: str,
                           user_id: int = Depends(verify_token)):
    left = get_owned_world_engine(world_id, left_branch, user_id)
    right = get_owned_world_engine(world_id, right_branch, user_id)
    left_snapshots = {item["month"]: item["state"] for item in list_world_snapshots(world_id, left_branch)}
    right_snapshots = {item["month"]: item["state"] for item in list_world_snapshots(world_id, right_branch)}
    months = sorted(set(left_snapshots) | set(right_snapshots))
    timeline = []
    for month in months:
        left_state, right_state = left_snapshots.get(month), right_snapshots.get(month)
        left_company = left_state["companies"]["player"] if left_state else None
        right_company = right_state["companies"]["player"] if right_state else None
        timeline.append({
            "month": month,
            "left_cash": left_company["cash"] if left_company else None,
            "right_cash": right_company["cash"] if right_company else None,
            "left_revenue": left_company["revenue"] if left_company else None,
            "right_revenue": right_company["revenue"] if right_company else None,
            "left_customers": left_company["customers"] if left_company else None,
            "right_customers": right_company["customers"] if right_company else None,
            "cash_delta": (right_company["cash"] - left_company["cash"]) if left_company and right_company else None,
            "revenue_delta": (right_company["revenue"] - left_company["revenue"]) if left_company and right_company else None,
            "customer_delta": (right_company["customers"] - left_company["customers"]) if left_company and right_company else None,
        })
    left_company, right_company = left.state.companies["player"], right.state.companies["player"]
    branches = {item["id"]: item for item in list_branches(world_id)}
    def decisions(branch_id):
        return [{"month": item["month"], "action": item["payload"].get("action")}
                for item in list_events(world_id, branch_id)
                if item["type"] == "company_action" and item["actor_id"] == "player"]
    return {
        "left": {"branch": branches.get(left_branch), "state": left.state.to_dict(), "decisions": decisions(left_branch)},
        "right": {"branch": branches.get(right_branch), "state": right.state.to_dict(), "decisions": decisions(right_branch)},
        "deltas": {
            "cash": right_company.cash - left_company.cash,
            "revenue": right_company.revenue - left_company.revenue,
            "customers": right_company.customers - left_company.customers,
            "product_quality": right_company.product_quality - left_company.product_quality,
            "founder_ownership": right_company.founder_ownership - left_company.founder_ownership,
        },
        "timeline": timeline,
        "interpretation": "Positive deltas mean the right branch is ahead of the left branch on that metric.",
    }


@app.post("/worlds/{world_id}/branches/{branch_id}/generate-trajectories")
def generate_world_trajectories(world_id: str, branch_id: str, request: GenerateTrajectoriesRequest,
                                user_id: int = Depends(verify_token)):
    if request.action not in ACTION_TYPES:
        raise HTTPException(status_code=422, detail="Unknown action")
    if not 1 <= request.horizon <= 36 or not 20 <= request.paths <= 1000:
        raise HTTPException(status_code=422, detail="Horizon must be 1-36 and paths 20-1000")
    engine = get_owned_world_engine(world_id, branch_id, user_id)
    return generate_trajectories(engine.state, request.action, request.horizon, request.paths, request.seed)


@app.get("/worlds/{world_id}/branches/{branch_id}/causal-effects")
def get_world_causal_effects(world_id: str, branch_id: str, user_id: int = Depends(verify_token)):
    engine = get_owned_world_engine(world_id, branch_id, user_id)
    return estimate_action_effects(engine.state)


@app.post("/worlds/{world_id}/branches/{branch_id}/model-based-ceo")
def get_model_based_ceo(world_id: str, branch_id: str, request: ModelBasedPlanRequest,
                        user_id: int = Depends(verify_token)):
    if not 2 <= request.horizon <= 24 or not 3 <= request.beam_width <= 30 or not 20 <= request.paths <= 200:
        raise HTTPException(status_code=422, detail="Invalid planning horizon, beam width, or path count")
    if not 0 <= request.risk_aversion <= 1:
        raise HTTPException(status_code=422, detail="Risk aversion must be between 0 and 1")
    engine = get_owned_world_engine(world_id, branch_id, user_id)
    return plan_actions(engine.state, request.horizon, request.beam_width, request.paths,
                        request.risk_aversion, request.seed)


@app.post("/worlds/{world_id}/branches/{branch_id}/compare-human-ai")
def compare_human_ai(world_id: str, branch_id: str, request: HumanAiComparisonRequest,
                     user_id: int = Depends(verify_token)):
    if request.human_action not in ACTION_TYPES:
        raise HTTPException(status_code=422, detail="Unknown human action")
    if not 0 <= request.risk_aversion <= 1:
        raise HTTPException(status_code=422, detail="Risk aversion must be between 0 and 1")
    parent = get_owned_world_engine(world_id, branch_id, user_id)
    plan = plan_actions(parent.state, horizon=12, beam_width=8, paths=40,
                        risk_aversion=request.risk_aversion, seed=request.seed)
    ai_action = plan["recommendation"]["first_action"]
    suffix = secrets.token_hex(3); month = parent.state.month
    human = parent.branch(month, f"human-{month}-{suffix}")
    ai = parent.branch(month, f"ai-{month}-{suffix}")
    create_branch_record(user_id, parent, human, f"Human: {request.human_action}", month)
    create_branch_record(user_id, parent, ai, f"AI: {ai_action}", month)
    human_state, human_events = human.advance(request.human_action)
    ai_state, ai_events = ai.advance(ai_action)
    persist_advance(human, human_events); persist_advance(ai, ai_events)
    return {"fork_month": month, "human_action": request.human_action, "ai_action": ai_action,
            "human_branch": human_state.to_dict(), "ai_branch": ai_state.to_dict(), "ai_plan": plan}


@app.get("/datasets")
def get_datasets(user_id: int = Depends(verify_token)):
    ensure_data_storage()
    return list_datasets(user_id)


@app.get("/ml/registry")
def get_model_registry(user_id: int = Depends(verify_token)):
    return model_registry()


@app.get("/datasets/{import_id}/observations")
def get_dataset_observations(import_id: int, limit: int = 1000, user_id: int = Depends(verify_token)):
    ensure_data_storage()
    return dataset_observations(user_id, import_id, min(max(limit, 1), 5000))


@app.post("/datasets/import/official")
def import_official_dataset(request: OfficialDataImportRequest, user_id: int = Depends(verify_token)):
    ensure_data_storage()
    if request.source == "fred":
        raw, observations, url = fetch_fred_macro(); name = "FRED macroeconomic indicators"
    elif request.source == "census_bds":
        raw, observations, url = fetch_census_business_dynamics(request.start_year); name = "Census Business Dynamics Statistics"
    elif request.source == "sec_companyfacts":
        if not request.cik: raise HTTPException(status_code=422, detail="CIK is required for SEC data")
        raw, observations, url = fetch_sec_companyfacts(request.cik); name = f"SEC Company Facts CIK {request.cik}"
    else:
        raise HTTPException(status_code=422, detail="Source must be fred, census_bds, or sec_companyfacts")
    return save_dataset(user_id, request.source, name, url, raw, observations,
                        {"connector": "official_api", "requested_start_year": request.start_year})


@app.post("/datasets/import/csv")
def import_csv_dataset(request: CsvDataImportRequest, user_id: int = Depends(verify_token)):
    ensure_data_storage()
    try:
        observations = parse_long_csv(request.csv_text)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return save_dataset(user_id, "user_csv", request.dataset_name, None, request.csv_text,
                        observations, {"connector": "long_format_csv"})
