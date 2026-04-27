"""
Example NKIPy kernel: Embedding lookup with bias

Gathers rows from an embedding table using np.take, then adds a bias.
This exercises the nkipy.gather op (lowered to nisa.dma_copy_indirect).
"""
import numpy as np
from nkipy_kernelgen import trace, knob

VOCAB = 1024
EMBED = 512
SEQ = 256

@trace(input_specs=[((VOCAB, EMBED), "f32"), ((SEQ,), "i32"), ((SEQ, EMBED), "f32")])
def embedding_lookup(table, token_ids, bias):
    
    gathered = np.take(table, token_ids, axis=0)
    knob.knob(gathered, mem_space="Sbuf", tile_size=[128, 128])

    result = np.add(gathered, bias)
    knob.knob(result, mem_space="SharedHbm", tile_size=[128, 128])

    return result
