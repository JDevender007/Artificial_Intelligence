from rlinv.envs import ENV_ID, InventoryEnv, InventoryParams
from rlinv.evaluate import evaluate_policy, mean_score
from rlinv.heuristics import base_stock, dow_levels, dow_sS, nearest_action, random_policy, sS, tune


def test_nearest_action():
    sizes = (0, 5, 10, 15)
    assert nearest_action(-3, sizes) == 0 and nearest_action(7, sizes) == 1 and nearest_action(8, sizes) == 2
    assert nearest_action(100, sizes) == 3


def test_base_stock_orders_up_to_position():
    env = InventoryEnv()
    env.reset(seed=0, options={"on_hand": 10, "pipeline": [5, 5], "weekday": 0})
    assert env.p.order_sizes[base_stock(40)(None, env)] == 20  # position 20, so order 20
    assert sS(15, 40)(None, env) == 0  # position 20 >= s


def test_dow_levels_follow_seasonality():
    lv = dow_levels(InventoryParams(), 0.9)
    assert lv.index(max(lv)) in (4, 5)  # windows starting Fri/Sat contain the weekend peak
    env = InventoryEnv()
    env.reset(seed=0, options={"on_hand": 0, "weekday": 4})
    assert dow_sS(0.9, 10)(None, env) > 0


def test_tune_picks_best_of_grid_and_beats_random():
    seeds = list(range(5))
    params, score = tune("base_stock", seeds, grid={"S": [0, 40]})
    assert params == {"S": 40}
    assert score > mean_score(evaluate_policy(ENV_ID, random_policy(0), seeds))
