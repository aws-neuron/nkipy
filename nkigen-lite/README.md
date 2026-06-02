# nkigen-lite

Lightweight IR-based kernel generation backend for NKIPy.

Provides a tensor-level IR (`tensor_ir`) and tile-level NKI IR (`nki_ir`) with
lowering passes to convert high-level tensor operations into NeuronCore-native
tile operations.
