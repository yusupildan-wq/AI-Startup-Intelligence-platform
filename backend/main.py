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
from ml.digital_twin import predict_digital_twin

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


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/model-metrics")
def model_metrics():
    return {
        "churn_model_comparison": _churn_model_benchmark,
        "growth_model": _growth_model_metrics,
        "fundraising_model": _fundraising_model_metrics,
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
