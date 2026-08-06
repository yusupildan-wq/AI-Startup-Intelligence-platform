import hashlib
import json

import psycopg2.extras

from state_store import get_connection


def ensure_data_tables():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS dataset_imports (
                id BIGSERIAL PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
                source TEXT NOT NULL, dataset_name TEXT NOT NULL, source_url TEXT,
                content_sha256 TEXT NOT NULL, row_count INTEGER NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                imported_at TIMESTAMP NOT NULL DEFAULT now())""")
            cur.execute("""CREATE TABLE IF NOT EXISTS external_observations (
                id BIGSERIAL PRIMARY KEY, import_id BIGINT NOT NULL REFERENCES dataset_imports(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL REFERENCES users(id), source TEXT NOT NULL,
                series TEXT NOT NULL, observation_date TEXT NOT NULL, value DOUBLE PRECISION,
                unit TEXT, entity TEXT, dimensions JSONB NOT NULL DEFAULT '{}'::jsonb,
                UNIQUE (import_id, series, observation_date, entity))""")
            # Migrate the first development schema, which moved observations
            # between manifests on re-import and therefore broke replayability.
            cur.execute("ALTER TABLE external_observations DROP CONSTRAINT IF EXISTS external_observations_user_id_source_series_observation_dat_key")
            cur.execute("ALTER TABLE external_observations DROP CONSTRAINT IF EXISTS external_observations_user_id_source_series_observatio_key")
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_external_observation_import ON external_observations(import_id, series, observation_date, entity)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_external_observations_lookup ON external_observations(user_id, source, series, observation_date)")


def content_hash(raw):
    if not isinstance(raw, bytes): raw = raw.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def save_dataset(user_id, source, dataset_name, source_url, raw_content, observations, metadata=None):
    digest = content_hash(raw_content)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO dataset_imports
                (user_id, source, dataset_name, source_url, content_sha256, row_count, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (user_id, source, dataset_name, source_url, digest, len(observations),
                 psycopg2.extras.Json(metadata or {})))
            import_id = cur.fetchone()[0]
            psycopg2.extras.execute_values(cur, """INSERT INTO external_observations
                (import_id, user_id, source, series, observation_date, value, unit, entity, dimensions)
                VALUES %s ON CONFLICT (import_id, series, observation_date, entity)
                DO UPDATE SET value=EXCLUDED.value, unit=EXCLUDED.unit,
                    dimensions=EXCLUDED.dimensions""",
                [(import_id, user_id, source, item["series"], item["date"], item.get("value"),
                  item.get("unit"), item.get("entity", ""), json.dumps(item.get("dimensions", {})))
                 for item in observations], template="(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)")
    return {"import_id": import_id, "source": source, "dataset_name": dataset_name,
            "row_count": len(observations), "content_sha256": digest}


def list_datasets(user_id):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM dataset_imports WHERE user_id=%s ORDER BY imported_at DESC", (user_id,))
            return [dict(row) for row in cur.fetchall()]


def dataset_observations(user_id, import_id, limit=1000):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""SELECT source, series, observation_date, value, unit, entity, dimensions
                FROM external_observations WHERE user_id=%s AND import_id=%s
                ORDER BY observation_date, series LIMIT %s""", (user_id, import_id, limit))
            return [dict(row) for row in cur.fetchall()]
