from nkipy_serving.runtime.precompile_catalog import (
    CompiledVariant,
    PrecompileCatalog,
    VariantKey,
)


class VariantRegistry:
    """Deterministic lookup table for compiled variants."""

    def __init__(self, catalog: PrecompileCatalog):
        self._variants: dict[VariantKey, CompiledVariant] = {}
        for variant in catalog.variants:
            if variant.key in self._variants:
                raise RuntimeError(f"Duplicate variant in registry: {variant.key}")
            self._variants[variant.key] = variant

    def resolve(self, key: VariantKey) -> CompiledVariant:
        variant = self._variants.get(key)
        if variant is None:
            raise RuntimeError(f"Variant not found in registry: {key}")
        return variant
