"""PostgreSQL event store for durable, branchable civilization worlds."""

import psycopg2.extras

from state_store import get_connection
from world.engine import WorldEngine
from world.events import WorldEvent
from world.models import world_from_dict


def ensure_world_tables():
    statements = (
        """CREATE TABLE IF NOT EXISTS simulation_worlds (
            id TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), name TEXT NOT NULL,
            seed INTEGER NOT NULL, created_at TIMESTAMP NOT NULL DEFAULT now())""",
        """CREATE TABLE IF NOT EXISTS world_branches (
            id TEXT NOT NULL, world_id TEXT NOT NULL REFERENCES simulation_worlds(id) ON DELETE CASCADE,
            parent_branch_id TEXT, fork_month INTEGER NOT NULL DEFAULT 0, name TEXT NOT NULL,
            initial_state JSONB NOT NULL, current_state JSONB NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT now(), PRIMARY KEY (world_id, id))""",
        """CREATE TABLE IF NOT EXISTS world_events (
            sequence BIGSERIAL PRIMARY KEY, id TEXT NOT NULL UNIQUE,
            world_id TEXT NOT NULL REFERENCES simulation_worlds(id) ON DELETE CASCADE,
            branch_id TEXT NOT NULL, month INTEGER NOT NULL, type TEXT NOT NULL,
            actor_id TEXT NOT NULL, payload JSONB NOT NULL, created_at TIMESTAMP NOT NULL DEFAULT now(),
            FOREIGN KEY (world_id, branch_id) REFERENCES world_branches(world_id, id) ON DELETE CASCADE)""",
        """CREATE TABLE IF NOT EXISTS world_snapshots (
            world_id TEXT NOT NULL, branch_id TEXT NOT NULL, month INTEGER NOT NULL,
            state JSONB NOT NULL, created_at TIMESTAMP NOT NULL DEFAULT now(),
            PRIMARY KEY (world_id, branch_id, month),
            FOREIGN KEY (world_id, branch_id) REFERENCES world_branches(world_id, id) ON DELETE CASCADE)""",
    )
    with get_connection() as conn:
        with conn.cursor() as cur:
            for statement in statements:
                cur.execute(statement)


def create_world_record(user_id, engine):
    state = engine.state
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO simulation_worlds (id, user_id, name, seed) VALUES (%s, %s, %s, %s)",
                        (state.id, user_id, state.name, state.seed))
            cur.execute("""INSERT INTO world_branches
                (id, world_id, parent_branch_id, fork_month, name, initial_state, current_state)
                VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (state.branch_id, state.id, None, 0, state.branch_id,
                 psycopg2.extras.Json(engine.initial_state.to_dict()), psycopg2.extras.Json(state.to_dict())))
            _insert_snapshot(cur, state.id, state.branch_id, state.month, state.to_dict())


def list_worlds(user_id):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""SELECT w.id, w.name, w.seed, w.created_at,
                COUNT(b.id) AS branch_count, MAX((b.current_state->>'month')::int) AS latest_month
                FROM simulation_worlds w JOIN world_branches b ON b.world_id = w.id
                WHERE w.user_id = %s GROUP BY w.id ORDER BY w.created_at DESC""", (user_id,))
            return [dict(row) for row in cur.fetchall()]


def assert_world_owner(world_id, user_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM simulation_worlds WHERE id = %s AND user_id = %s", (world_id, user_id))
            return cur.fetchone() is not None


def load_engine(world_id, branch_id):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM world_branches WHERE world_id = %s AND id = %s", (world_id, branch_id))
            branch = cur.fetchone()
            if not branch:
                return None
            engine = WorldEngine(world_from_dict(branch["initial_state"]))
            cur.execute("SELECT * FROM world_events WHERE world_id = %s AND branch_id = %s ORDER BY sequence",
                        (world_id, branch_id))
            for row in cur.fetchall():
                engine.apply(WorldEvent(row["month"], row["type"], row["actor_id"], row["payload"], row["id"]))
            cur.execute("SELECT month, state FROM world_snapshots WHERE world_id = %s AND branch_id = %s ORDER BY month",
                        (world_id, branch_id))
            engine.snapshots = {row["month"]: world_from_dict(row["state"]) for row in cur.fetchall()}
            engine.state = world_from_dict(branch["current_state"])
            return engine


def persist_advance(engine, new_events):
    state = engine.state
    with get_connection() as conn:
        with conn.cursor() as cur:
            for event in new_events:
                cur.execute("""INSERT INTO world_events (id, world_id, branch_id, month, type, actor_id, payload)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (event.id, state.id, state.branch_id, event.month, event.type, event.actor_id,
                     psycopg2.extras.Json(event.payload)))
            _insert_snapshot(cur, state.id, state.branch_id, state.month, state.to_dict())
            cur.execute("UPDATE world_branches SET current_state = %s WHERE world_id = %s AND id = %s",
                        (psycopg2.extras.Json(state.to_dict()), state.id, state.branch_id))


def create_branch_record(user_id, parent_engine, branch_engine, branch_name, fork_month):
    if not assert_world_owner(parent_engine.state.id, user_id):
        raise PermissionError("Not your world")
    state = branch_engine.state
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO world_branches
                (id, world_id, parent_branch_id, fork_month, name, initial_state, current_state)
                VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (state.branch_id, state.id, state.parent_branch_id, fork_month, branch_name,
                 psycopg2.extras.Json(state.to_dict()), psycopg2.extras.Json(state.to_dict())))
            _insert_snapshot(cur, state.id, state.branch_id, state.month, state.to_dict())


def list_branches(world_id):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""SELECT id, parent_branch_id, fork_month, name,
                (current_state->>'month')::int AS current_month, created_at
                FROM world_branches WHERE world_id = %s ORDER BY created_at""", (world_id,))
            return [dict(row) for row in cur.fetchall()]


def list_events(world_id, branch_id):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""SELECT id, month, type, actor_id, payload, sequence
                FROM world_events WHERE world_id = %s AND branch_id = %s ORDER BY sequence""",
                (world_id, branch_id))
            return [dict(row) for row in cur.fetchall()]


def _insert_snapshot(cur, world_id, branch_id, month, state):
    cur.execute("""INSERT INTO world_snapshots (world_id, branch_id, month, state)
        VALUES (%s, %s, %s, %s) ON CONFLICT (world_id, branch_id, month)
        DO UPDATE SET state = EXCLUDED.state""",
        (world_id, branch_id, month, psycopg2.extras.Json(state)))
