import secrets

import bcrypt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from state_store import insert_startup, get_startup, get_all_snapshots, insert_user, get_user_by_email
from orchestrator import run_month
from auth import create_token, verify_token
from prediction_engine import benchmark_churn_models, train_growth_model

app = FastAPI()

_churn_model_benchmark = benchmark_churn_models()
_, _growth_model_metrics = train_growth_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/model-metrics")
def model_metrics():
    return {
        "churn_model_comparison": _churn_model_benchmark,
        "growth_model": _growth_model_metrics,
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


@app.post("/startups/{startup_id}/simulate-next-month")
def simulate_next_month(startup_id: int, request: SimulateMonthRequest, user_id: int = Depends(verify_token)):
    get_owned_startup_or_403(startup_id, user_id)

    return run_month(
        startup_id=startup_id,
        marketing_spend=request.marketing_spend,
        employee_count=request.employee_count,
    )
