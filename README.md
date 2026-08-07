# Startup Civilization Lab

An experimental multi-agent environment for evaluating startup decision policies under uncertainty.

The central question is deliberately narrow:

> Can a CEO that plans through a learned world model outperform simpler controllers when every policy starts from the same unseen civilization?

This repository is not presented as a validated predictor of real startup outcomes. It is a reproducible ML/simulation research project with an explicit synthetic-domain limitation.

## Central benchmark

Five controllers were evaluated for 12 months on identical copies of 24 held-out generated civilizations:

| Controller | Survival | Median founder-value proxy | Stress cash (P10) | Median ownership | Median fundraises |
|---|---:|---:|---:|---:|---:|
| Model-based CEO | 100.0% | $3.60M | $9.72M | 11.7% | 11 |
| Transferred RL CEO | 100.0% | $3.74M | $0.90M | 86.9% | 1 |
| Runway heuristic | 95.8% | $1.74M | $0.14M | 90.0% | 0 |
| Always hold | 100.0% | $1.69M | $0.42M | 90.0% | 0 |
| Random actions | 95.8% | $2.85M | $0.53M | 67.5% | 1 |

### Finding

The model-based CEO maximized survival and cash by fundraising a median 11 times. After accounting for dilution, it did **not** beat the transferred RL CEO on median founder value. This exposed reward misspecification: the planner's learned state and objective rewarded access to synthetic capital without adequately valuing retained ownership.

That negative result is the project's primary result. It demonstrates why survival or cash alone is an insufficient policy objective and motivates adding ownership to the trajectory model before claiming that model-based planning is superior.

Reproduce the benchmark:

```bash
cd backend
python -m ml.evaluate_central_benchmark
```

The versioned result is stored in `backend/models/central_benchmark_v1_metrics.json`.

## Evaluation design

- **Matched worlds:** every controller receives an exact copy of each starting civilization.
- **Held-out seeds:** benchmark seeds are separate from transition-model training worlds.
- **True-environment evaluation:** policies plan with learned models but are evaluated afterward by the event-driven civilization engine.
- **Uncertainty:** survival and median founder value include 95% bootstrap intervals.
- **Founder-adjusted objective:** founder value is the disclosed enterprise-value proxy multiplied by retained ownership.
- **Baselines:** transferred offline RL, a runway heuristic, always hold, and random actions.

The founder-value proxy is:

```text
(cash + 12 × monthly revenue + $500 × customers + $100,000 × product quality)
× retained founder ownership
```

It is an experimental simulator metric, not a financial valuation.

## Technical system

The benchmark is supported by a larger research environment:

- Event-driven, replayable civilization engine
- PostgreSQL event, branch, and snapshot persistence
- Learned customer-choice, employee-attrition, and product-adoption models
- Learned investor, competitor-policy, and macro-regime agents
- Probabilistic full-world generator
- Action-conditioned generative trajectory model
- Paired-counterfactual causal action model
- Offline fitted-Q reinforcement-learning CEO
- Beam-search model-based CEO with stochastic stress testing
- Persistent human-versus-AI forks and branch-difference viewer
- FRED, Census BDS, SEC Company Facts, and longitudinal CSV ingestion
- Cryptographic dataset and model lineage registry
- React/TypeScript recruiter-facing experiment interface

## Architecture

```text
Generated unseen world
        │
        ├── Model-based planner ── learned transition model
        ├── Transferred RL CEO ─── fitted Q policy
        ├── Runway heuristic
        ├── Hold baseline
        └── Random baseline
                    │
                    ▼
         Event-driven world engine
                    │
                    ▼
      Matched survival, cash, value,
      ownership and policy comparison
```

The application also contains a connected legacy company workflow for monthly simulation, Digital Twin forecasting, strategy search, and OpenAI narration. These tools are supporting demonstrations; the matched controller benchmark is the central ML contribution.

## Run locally

Requirements:

- Python 3.11+
- Node.js 20+
- PostgreSQL
- An OpenAI API key for the narration feature

Create the database and apply the schema:

```bash
createdb -U postgres startup_intel
psql -U postgres -d startup_intel -f database/schema.sql
```

The development database configuration is currently in `backend/state_store.py`.

Start the backend:

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
set OPENAI_API_KEY=your_key_here
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Start the frontend in another terminal:

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173` and choose **Try as Guest**.

## Tests

```bash
cd backend
python -m pytest -q

cd ../frontend
npm run build
```

## Limitations

- The civilization and its benchmark are synthetic.
- Real Data Lab imports do not automatically retrain the simulator models.
- The current trajectory model omits founder ownership, which the benchmark identified as a material planning flaw.
- Twenty-four worlds are sufficient for a reproducible project benchmark, not broad scientific certainty.
- Performance inside this environment does not imply real-world investment or operating performance.

## Development ownership

AI tools accelerated implementation, but the project is organized around inspectable engineering and experimental decisions: matched evaluation, held-out worlds, baseline selection, artifact lineage, metric correction, and documented failure analysis. Every model exposes its training source, reproduction command, metrics, and limitations so the work can be questioned rather than accepted as a black box.
