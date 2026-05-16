import click

from kaggle_otto2.cand_generator import ItemCFCandGenerator
from kaggle_otto2.cand_generator.item_cf.params import (
    ITEMCF_AGG_METHODS,
    get_itemcf_params,
)
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader


@click.command()
@click.option("--exp", required=True, type=str)
def main(exp: str):
    config = Config(exp)
    data_loader = OttoDataLoader(config)
    params = get_itemcf_params(config)

    print(f"ItemCF param groups: {len(params)}")
    print(f"ItemCF agg methods: {ITEMCF_AGG_METHODS}")

    for param in params:
        cg = ItemCFCandGenerator(
            config.dir_config.exp_output_dir,
            data_loader,
            config,
            "sum",
            *param,
        )
        cg.fit()

    for param in params:
        for agg_method in ITEMCF_AGG_METHODS:
            cg = ItemCFCandGenerator(
                config.dir_config.exp_output_dir,
                data_loader,
                config,
                agg_method,
                *param,
            )
            cg.gen_cand_df()
            if config.is_cv:
                cg.calc_score()


if __name__ == "__main__":
    main()
