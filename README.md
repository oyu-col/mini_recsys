# mini_recsys

Kaggle-ready mini version of the `kiccho1101/kaggle-otto2` OTTO recommender
pipeline. The core two-stage logic is preserved:

1. preprocess local validation data
2. generate candidates with last interaction, ItemCF, ItemMF, UserMF, and Item2Vec
3. merge candidates
4. build features and negative samples
5. train LightGBM rankers and report CV recall

The code is adapted for Kaggle Notebook execution:

- reads datasets from `/kaggle/input`
- writes outputs to `/kaggle/working/output`
- auto-detects a CV dataset containing `train.parquet`, `test.parquet`, and
  `test_labels.parquet`
- keeps compatibility with newer Polars APIs
- uses CUDA when available and falls back to CPU
- treats CatBoost/XGBoost as optional dependencies
- keeps Kaggle's preinstalled NumPy/Pandas/SciPy/Torch stack intact

## Kaggle 5% CV Run

Add the Kaggle dataset `OTTO train and validation (extracted from train)` to the
Notebook sidebar. If the dataset directory is not auto-detected, set:

```bash
export OTTO_CV_DATASET_DIR=/kaggle/input/<your-dataset-slug>
```

Then run:

```bash
pip install -r requirements-kaggle.txt

PYTHONPATH=. python kaggle_otto2/data_loader/main.py --exp exp001_dev
PYTHONPATH=. python kaggle_otto2/cand_generator/last_inter/main.py --exp exp001_dev
PYTHONPATH=. python kaggle_otto2/cand_generator/item_cf/main.py --exp exp001_dev
PYTHONPATH=. python kaggle_otto2/cand_generator/item_mf/main.py --exp exp001_dev
PYTHONPATH=. python kaggle_otto2/cand_generator/user_mf/main.py --exp exp001_dev
PYTHONPATH=. python kaggle_otto2/cand_generator/item2vec/main.py --exp exp001_dev
PYTHONPATH=. python kaggle_otto2/cand_merger/main.py --exp exp001_dev
PYTHONPATH=. python kaggle_otto2/feature/main.py --exp exp001_dev
PYTHONPATH=. python kaggle_otto2/ranker_trainer/main.py --exp exp001_dev --model_type lgbm
```

You can also upload and run `otto_mini.ipynb` on Kaggle. It clones this
repository, detects the mounted CV dataset, and runs the same `exp001_dev`
pipeline.

For a first smoke test, reduce MF epochs in `yaml/exp001_dev.yaml`.

## Upstream

Based on the 20th place OTTO solution by `kiccho1101/kaggle-otto2`. The original
README is preserved as `README_UPSTREAM.md`.
