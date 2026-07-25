"""DeepSeek-V4 runtime assembly internals.

Module roles:

* ``surface`` builds the immutable runtime surface from loaded device weights.
* ``topology`` owns rank/group topology helpers used during assembly.
* ``install`` builds runtime components from the surface, graph functions,
  logits processor, and device state.
"""
