from ml.causal_actions import estimate_action_effects
from ml.model_based_ceo import plan_actions
from ml.world_generator import generate_learned_world
from world.events import ACTION_TYPES


def test_causal_model_compares_every_action_to_hold():
    world = generate_learned_world("causal-test", 71)
    result = estimate_action_effects(world)
    assert {item["action"] for item in result["effects"]} == set(ACTION_TYPES)
    hold = next(item for item in result["effects"] if item["action"] == "hold")
    assert all(value == 0 for value in hold["effects_vs_hold"].values())


def test_model_based_ceo_searches_and_ranks_every_first_action():
    world = generate_learned_world("planner-test", 72)
    result = plan_actions(world, horizon=2, beam_width=3, paths=20, seed=73)
    assert len(result["action_comparison"]) == len(ACTION_TYPES)
    assert result["recommendation"]["rank"] == 1
    assert len(result["recommendation"]["planned_sequence"]) == 2
