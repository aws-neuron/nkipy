import sys

import ml_dtypes
import numpy as np
import torch.distributed as dist

bfloat16 = np.dtype(ml_dtypes.bfloat16)


def print_log(msg, rank_list=[0], verbose=0):
    if not dist.is_initialized():
        print(msg)
    elif dist.get_rank() in rank_list:
        print(f"[RANK {dist.get_rank()}] {msg}")
        sys.stdout.flush()


def encode_prompt(tokenizer, prompt, raw=False):
    """Tokenize ``prompt`` to input ids of shape (1, L).

    gpt-oss is an instruct model, so by default we wrap the prompt in the chat
    template (``add_generation_prompt=True``). This matches the P-EAGLE
    drafter's training distribution; raw completion prompts are out of
    distribution and substantially lower acceptance length. Pass ``raw=True``
    (or use a tokenizer without a chat template) to tokenize the raw text.
    """
    if raw or tokenizer.chat_template is None:
        return tokenizer(prompt, return_tensors="np")["input_ids"]
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="np",
        return_dict=True,
    )["input_ids"]
