from world import WorldEngine, create_world


def test_world_advances_all_companies_and_records_events():
    engine = WorldEngine(create_world(seed=10))
    state, events = engine.advance("hire_engineer")
    assert state.month == 1
    assert state.companies["player"].engineers == 4
    assert state.companies["player"].revenue > 0
    assert len(events) >= 6


def test_same_seed_and_actions_are_deterministic():
    first = WorldEngine(create_world(seed=44))
    second = WorldEngine(create_world(seed=44))
    for action in ("raise_price", "hire_sales", "invest_in_product"):
        first.advance(action); second.advance(action)
    first.state.id = second.state.id
    assert first.state.to_dict() == second.state.to_dict()


def test_branch_is_independent_and_supports_shocks():
    main = WorldEngine(create_world(seed=81))
    main.advance("hold")
    branch = main.branch(1, "recession-case")
    main.advance("increase_marketing")
    branch.advance("decrease_marketing", shock="recession")
    assert branch.state.branch_id == "recession-case"
    assert branch.state.macro.regime == "recession"
    assert branch.state.companies["player"].marketing != main.state.companies["player"].marketing


def test_event_log_replays_to_identical_state():
    engine = WorldEngine(create_world(seed=19))
    engine.advance("raise_price")
    engine.advance("hire_support", shock="demand_surge")
    replayed = engine.replay()
    assert replayed.to_dict() == engine.state.to_dict()
