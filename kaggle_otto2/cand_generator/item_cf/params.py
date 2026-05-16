ITEMCF_AGG_METHODS = ["sum", "last", "count"]

# [source_types, target_types, use_aid_pop, use_ts_diff, use_trend, use_iif, use_future]
ALL_ITEMCF_PARAMS = [
    [[0, 1, 2], [0, 1, 2], True, True, False, True, False],
    [[1, 2], [1, 2], True, True, False, True, False],
    [[0], [0], True, True, False, True, False],
    [[0], [1, 2], True, True, False, True, False],
    [[0], [0, 1, 2], True, True, False, True, False],
    [[1, 2], [0, 1, 2], True, True, False, True, False],
    [[0, 1, 2], [0, 1, 2], True, True, True, True, True],
    [[1, 2], [0, 1, 2], True, True, True, True, True],
    [[0, 1, 2], [0, 1, 2], False, True, True, True, False],
]

CORE3_ITEMCF_PARAMS = [
    [[0, 1, 2], [0, 1, 2], True, True, False, True, False],
    [[1, 2], [1, 2], True, True, False, True, False],
    [[0], [0, 1, 2], True, True, False, True, False],
]


def get_itemcf_params(config):
    item_cf_cfg = config.yaml.cg.get("item_cf", {})
    preset = item_cf_cfg.get("param_preset", "all")
    if preset == "core3":
        return CORE3_ITEMCF_PARAMS
    if preset == "all":
        return ALL_ITEMCF_PARAMS
    raise ValueError(f"Unsupported item_cf.param_preset: {preset}")
