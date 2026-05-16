import click

from kaggle_otto2.cand_generator import (
    Item2VecCandGenerator,
    ItemCFCandGenerator,
    ItemMFCandGenerator,
    LastInterCandGenerator,
    UserMFCandGenerator,
)
from kaggle_otto2.cand_generator.item_cf.params import (
    ITEMCF_AGG_METHODS,
    get_itemcf_params,
)
from kaggle_otto2.cand_merger import CandMerger
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader


@click.command()
@click.option("--exp", required=True, type=str)
def main(exp: str):
    config = Config(exp)
    data_loader = OttoDataLoader(config)

    common_params = [config.dir_config.exp_output_dir, config, data_loader]
    itemcf_params = get_itemcf_params(config)
    cand_generators = [
        # LastInter
        LastInterCandGenerator(*[*common_params, "inter"]),
        LastInterCandGenerator(*[*common_params, "buy"]),
        LastInterCandGenerator(*[*common_params, "click"]),
        LastInterCandGenerator(*[*common_params, "cart"]),
        LastInterCandGenerator(*[*common_params, "order"]),
        # ItemCF
        *[
            ItemCFCandGenerator(*[*common_params, agg_method, *itemcf_param])
            for itemcf_param in itemcf_params
            for agg_method in ITEMCF_AGG_METHODS
        ],
        # ItemMF
        ItemMFCandGenerator(*[*common_params, "last"]),
        ItemMFCandGenerator(*[*common_params, "seq"]),
        # UserMF
        UserMFCandGenerator(*common_params),
        # Item2Vec
        Item2VecCandGenerator(*[*common_params, "last"]),
        Item2VecCandGenerator(*[*common_params, "seq"]),
    ]
    cand_merger = CandMerger(config.dir_config.exp_output_dir, config, data_loader)
    cand_merger.merge(cand_generators)
    cand_merger.calc_score()


if __name__ == "__main__":
    main()
