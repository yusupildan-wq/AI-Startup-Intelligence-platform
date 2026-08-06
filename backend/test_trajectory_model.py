from ml.trajectory_model import generate_trajectories
from ml.world_generator import generate_learned_world


def test_trajectory_generator_returns_seeded_uncertainty_bands():
    world = generate_learned_world("Test", 55)
    first = generate_trajectories(world, "hold", horizon=3, paths=20, seed=7)
    second = generate_trajectories(world, "hold", horizon=3, paths=20, seed=7)
    assert first == second
    assert len(first["timeline"]) == 3
    for month in first["timeline"]:
        assert month["cash_p10"] <= month["cash_median"] <= month["cash_p90"]
        assert 0 <= month["survival_probability"] <= 1
