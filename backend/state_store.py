import psycopg2
import psycopg2.extras

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "startup_intel",
    "user": "postgres",
    "password": "devpassword",
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def insert_startup(name, business_type, initial_price, founder_count, initial_funding):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO startups (name, business_type, initial_price, founder_count, initial_funding)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (name, business_type, initial_price, founder_count, initial_funding),
            )
            startup_id = cur.fetchone()[0]
        conn.commit()
    return startup_id


def insert_monthly_snapshot(
    startup_id,
    month_number,
    cash_on_hand,
    customer_count,
    customers_churned,
    revenue,
    employee_count,
    investor_count,
    funding_raised_to_date,
    price_per_customer,
    marketing_spend,
):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO monthly_snapshots (
                    startup_id, month_number, cash_on_hand, customer_count,
                    customers_churned, revenue, employee_count, investor_count,
                    funding_raised_to_date, price_per_customer, marketing_spend
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    startup_id, month_number, cash_on_hand, customer_count,
                    customers_churned, revenue, employee_count, investor_count,
                    funding_raised_to_date, price_per_customer, marketing_spend,
                ),
            )
        conn.commit()


def get_latest_snapshot(startup_id):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM monthly_snapshots
                WHERE startup_id = %s
                ORDER BY month_number DESC
                LIMIT 1
                """,
                (startup_id,),
            )
            row = cur.fetchone()
    return dict(row) if row else None
