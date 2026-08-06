import bcrypt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from state_store import insert_startup, get_startup, get_all_snapshots, insert_user, get_user_by_email
from orchestrator import run_month
from auth import create_token

app = FastAPI()

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


@app.post("/startups")
def create_startup(request: CreateStartupRequest):
    startup_id = insert_startup(
        request.name,
        request.business_type,
        request.initial_price,
        request.founder_count,
        request.initial_funding,
        request.initial_customer_count,
    )
    return {"startup_id": startup_id}


@app.get("/startups/{startup_id}/snapshots")
def list_snapshots(startup_id: int):
    if get_startup(startup_id) is None:
        raise HTTPException(status_code=404, detail="Startup not found")

    return get_all_snapshots(startup_id)


@app.post("/startups/{startup_id}/simulate-next-month")
def simulate_next_month(startup_id: int, request: SimulateMonthRequest):
    if get_startup(startup_id) is None:
        raise HTTPException(status_code=404, detail="Startup not found")

    return run_month(
        startup_id=startup_id,
        marketing_spend=request.marketing_spend,
        employee_count=request.employee_count,
    )
