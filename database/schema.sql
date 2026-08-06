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
