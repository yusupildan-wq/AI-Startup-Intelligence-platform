import hashlib
import json
from functools import lru_cache
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BACKEND_DIR / "models"

MODEL_SPECS = {
    "digital_twin_v1": ("ml/train_digital_twin.py", "python -m ml.train_digital_twin"),
    "ai_ceo_v1": ("ml/train_ai_ceo.py", "python -m ml.train_ai_ceo"),
    "population_models_v1": ("ml/train_population_models.py", "python -m ml.train_population_models"),
    "economy_agents_v1": ("ml/train_economy_agents.py", "python -m ml.train_economy_agents"),
    "world_generator_v1": ("ml/train_world_generator.py", "python -m ml.train_world_generator"),
    "trajectory_model_v1": ("ml/train_trajectory_model.py", "python -m ml.train_trajectory_model"),
    "causal_actions_v1": ("ml/train_causal_actions.py", "python -m ml.train_causal_actions"),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def model_registry():
    records = []
    for model_id, (training_file, command) in MODEL_SPECS.items():
        artifact = MODEL_DIR / f"{model_id}.joblib"
        metrics_path = MODEL_DIR / f"{model_id}_metrics.json"
        source_path = BACKEND_DIR / training_file
        if not artifact.exists() or not metrics_path.exists() or not source_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        data_source = metrics.get("data_source", "synthetic_generator_defined_in_training_code")
        records.append({
            "model_id": model_id,
            "version": "v1",
            "status": "active",
            "artifact": artifact.name,
            "artifact_bytes": artifact.stat().st_size,
            "artifact_sha256": sha256_file(artifact),
            "metrics_sha256": sha256_file(metrics_path),
            "training_code": training_file,
            "training_code_sha256": sha256_file(source_path),
            "reproduce_command": command,
            "data_lineage": {
                "source": data_source,
                "kind": "synthetic" if "synthetic" in data_source or "generated" in data_source else "external",
                "generator_locked_by": sha256_file(source_path),
            },
            "metrics": metrics,
            "limitations": (
                "Validated inside a synthetic or generated environment; artifact metrics do not establish "
                "real-world causal accuracy. External dataset manifests must be attached before real-data claims."
            ),
        })
    return records
