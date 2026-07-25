"""Runtime catalog, shape-guard, and execution adapter primitives."""

from nkipy_serving.runtime.precompile_catalog import (
    CompiledVariant,
    PrecompileCatalog,
    VariantKey,
    load_precompile_catalog,
    validate_precompile_catalog,
)
from nkipy_serving.runtime.precompile_paddings import (
    PrecompilePaddings,
    build_precompile_paddings,
)
from nkipy_serving.runtime.shape_guard import (
    select_bucket,
    validate_forward_batch_shape,
)
from nkipy_serving.runtime.variant_registry import VariantRegistry

__all__ = [
    "CompiledVariant",
    "PrecompileCatalog",
    "VariantKey",
    "VariantRegistry",
    "PrecompilePaddings",
    "build_precompile_paddings",
    "load_precompile_catalog",
    "validate_precompile_catalog",
    "select_bucket",
    "validate_forward_batch_shape",
]
