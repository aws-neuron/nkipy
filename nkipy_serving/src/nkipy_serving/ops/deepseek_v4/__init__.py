"""DeepSeek-V4 model-specific raw NKI kernels.

Unlike ``nkipy_serving/ops/attention/`` which hosts reusable attention kernels, this
package holds kernels whose shapes/semantics are only meaningful in V4:
index construction, top-k metadata mutation, and DSV4 device-state consumers.
"""
