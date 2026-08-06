from ml.ai_ceo import recommend_action
from ml.ai_ceo_environment import ACTIONS, StartupState


def test_ai_ceo_returns_ranked_action_and_rollout():
    state = StartupState(500000, 200, 60, 5000, 3, 1, 1, 0.6, 0.55, 0.3, 0.85, 1.0)
    result = recommend_action(state, rollout_months=4)
    assert result["recommendation"]["action"] in ACTIONS
    assert len(result["alternatives"]) == 4
    assert 1 <= len(result["projected_trajectory"]) <= 4
    assert result["policy"]["policy"]["survival_rate"] > result["policy"]["random_baseline"]["survival_rate"]
