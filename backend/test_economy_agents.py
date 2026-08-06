from ml.economy_agents import competitor_action, investor_offer, macro_regime
from ml.train_economy_agents import COMPETITOR_ACTIONS, MACRO_REGIMES


def test_economy_agents_return_valid_decisions():
    probability, amount = investor_offer([.25, 100000, 500000, 18, .7, .6, .7, .04, .8])
    assert 0 <= probability <= 1 and amount > 0
    assert competitor_action([1.2, .2, 1.0, .1, .1, .3, 1.0, .6, .4, .5]) in COMPETITOR_ACTIONS
    assert macro_regime([1.0, .05, .045, .6, .03, .1, .1, .1]) in MACRO_REGIMES
