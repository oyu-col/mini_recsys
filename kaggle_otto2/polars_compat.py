"""Compatibility helpers for running the original OTTO code on newer Polars."""

from __future__ import annotations

import inspect
from functools import wraps

import polars as pl


def _wrap_sort(method):
    sig = inspect.signature(method)
    params = sig.parameters

    @wraps(method)
    def wrapper(self, by=None, *args, reverse=None, descending=None, **kwargs):
        if reverse is not None and descending is None:
            descending = reverse

        if "descending" in params:
            if descending is not None:
                kwargs["descending"] = descending
        elif "reverse" in params:
            if descending is not None:
                kwargs["reverse"] = descending

        if by is None:
            return method(self, *args, **kwargs)
        return method(self, by, *args, **kwargs)

    wrapper._otto_compat_patched = True
    return wrapper


def _wrap_rank(method):
    sig = inspect.signature(method)
    params = sig.parameters

    @wraps(method)
    def wrapper(self, *args, reverse=None, descending=None, **kwargs):
        if reverse is not None and descending is None:
            descending = reverse

        if "descending" in params:
            if descending is not None:
                kwargs["descending"] = descending
        elif "reverse" in params:
            if descending is not None:
                kwargs["reverse"] = descending

        return method(self, *args, **kwargs)

    wrapper._otto_compat_patched = True
    return wrapper


def patch_polars() -> None:
    for cls in (pl.DataFrame, pl.LazyFrame):
        if not hasattr(cls, "with_column") and hasattr(cls, "with_columns"):

            def with_column(self, expr):
                return self.with_columns(expr)

            setattr(cls, "with_column", with_column)

        if not hasattr(cls, "groupby") and hasattr(cls, "group_by"):
            setattr(cls, "groupby", cls.group_by)

        sort_method = getattr(cls, "sort", None)
        if sort_method is not None and not getattr(
            sort_method, "_otto_compat_patched", False
        ):
            setattr(cls, "sort", _wrap_sort(sort_method))

    if not hasattr(pl.Expr, "apply") and hasattr(pl.Expr, "map_elements"):

        def apply(self, function, return_dtype=None, skip_nulls=True, **kwargs):
            return self.map_elements(
                function,
                return_dtype=return_dtype,
                skip_nulls=skip_nulls,
                **kwargs,
            )

        setattr(pl.Expr, "apply", apply)

    rank_method = getattr(pl.Expr, "rank", None)
    if rank_method is not None and not getattr(
        rank_method, "_otto_compat_patched", False
    ):
        setattr(pl.Expr, "rank", _wrap_rank(rank_method))

    value_counts_method = getattr(pl.Series, "value_counts", None)
    if value_counts_method is not None and not getattr(
        value_counts_method, "_otto_compat_patched", False
    ):

        @wraps(value_counts_method)
        def value_counts(self, *args, **kwargs):
            df = value_counts_method(self, *args, **kwargs)
            if "count" in df.columns and "counts" not in df.columns:
                df = df.rename({"count": "counts"})
            return df

        value_counts._otto_compat_patched = True
        setattr(pl.Series, "value_counts", value_counts)


patch_polars()
