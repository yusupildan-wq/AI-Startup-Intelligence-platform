from ml.registry import MODEL_SPECS, model_registry


def test_registry_has_reproducible_hashes_for_every_model():
    records = model_registry()
    assert len(records) == len(MODEL_SPECS)
    assert all(len(record["artifact_sha256"]) == 64 for record in records)
    assert all(len(record["training_code_sha256"]) == 64 for record in records)
    assert all(record["reproduce_command"].startswith("python -m ml.") for record in records)
