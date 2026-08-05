from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from state_store import insert_startup, get_startup
from orchestrator import run_month

app = FastAPI()


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


@app.post("/startups/{startup_id}/simulate-next-month")
def simulate_next_month(startup_id: int, request: SimulateMonthRequest):
    if get_startup(startup_id) is None:
        raise HTTPException(status_code=404, detail="Startup not found")

    return run_month(
        startup_id=startup_id,
        marketing_spend=request.marketing_spend,
        employee_count=request.employee_count,
    )
