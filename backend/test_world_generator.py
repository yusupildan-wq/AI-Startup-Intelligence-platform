from ml.world_generator import generate_learned_world


def test_generator_is_seeded_valid_and_scenario_conditioned():
    first = generate_learned_world("Generated", 91)
    second = generate_learned_world("Generated", 91)
    first.id = second.id
    assert first.to_dict() == second.to_dict()
    assert len(first.companies) == 3 and len(first.segments) == 3
    recession = generate_learned_world("Downturn", 91, "recession")
    assert recession.macro.regime == "recession"
    assert recession.macro.demand_multiplier == .68
