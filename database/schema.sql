CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE startups (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    name TEXT NOT NULL,
    business_type TEXT NOT NULL,
    initial_price NUMERIC(10,2) NOT NULL,
    founder_count INTEGER NOT NULL,
    initial_funding NUMERIC(12,2) NOT NULL,
    initial_customer_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE monthly_snapshots (
    id SERIAL PRIMARY KEY,
    startup_id INTEGER NOT NULL REFERENCES startups(id),
    month_number INTEGER NOT NULL,
    cash_on_hand NUMERIC(12,2) NOT NULL,
    customer_count INTEGER NOT NULL,
    customers_churned INTEGER NOT NULL,
    customers_acquired INTEGER NOT NULL DEFAULT 0,
    revenue NUMERIC(12,2) NOT NULL,
    employee_count INTEGER NOT NULL,
    investor_count INTEGER NOT NULL,
    funding_raised_to_date NUMERIC(12,2) NOT NULL,
    price_per_customer NUMERIC(10,2) NOT NULL,
    marketing_spend NUMERIC(12,2) NOT NULL,
    UNIQUE (startup_id, month_number)
);

CREATE TABLE simulation_worlds (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    name TEXT NOT NULL,
    seed INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE world_branches (
    id TEXT NOT NULL,
    world_id TEXT NOT NULL REFERENCES simulation_worlds(id) ON DELETE CASCADE,
    parent_branch_id TEXT,
    fork_month INTEGER NOT NULL DEFAULT 0,
    name TEXT NOT NULL,
    initial_state JSONB NOT NULL,
    current_state JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (world_id, id)
);

CREATE TABLE world_events (
    sequence BIGSERIAL PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    world_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    month INTEGER NOT NULL,
    type TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    FOREIGN KEY (world_id, branch_id) REFERENCES world_branches(world_id, id) ON DELETE CASCADE
);

CREATE TABLE world_snapshots (
    world_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    month INTEGER NOT NULL,
    state JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (world_id, branch_id, month),
    FOREIGN KEY (world_id, branch_id) REFERENCES world_branches(world_id, id) ON DELETE CASCADE
);

CREATE TABLE dataset_imports (
    id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    source TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    source_url TEXT,
    content_sha256 TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    imported_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE external_observations (
    id BIGSERIAL PRIMARY KEY,
    import_id BIGINT NOT NULL REFERENCES dataset_imports(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id),
    source TEXT NOT NULL,
    series TEXT NOT NULL,
    observation_date TEXT NOT NULL,
    value DOUBLE PRECISION,
    unit TEXT,
    entity TEXT,
    dimensions JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (import_id, series, observation_date, entity)
);
