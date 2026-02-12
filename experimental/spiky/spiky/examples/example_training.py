# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training example using the spiky backend with dynamic shapes.

Trains a small MLP for several steps with varying batch sizes on Neuron
hardware and compares with eager CPU results.

Usage:
    cd nkipy/
    uv run python experimental/spiky/spiky/examples/example_training.py
"""

import copy

import torch
import torch.nn as nn
from spiky.torch import init_nkipy_backend

torch.manual_seed(0)

init_nkipy_backend(nkipy_cache="./nkipy_cache")

# Define model
model = nn.Sequential(
    nn.Linear(32, 64, bias=False),
    nn.ReLU(),
    nn.Linear(64, 16, bias=False),
)

# Eager reference (CPU)
ref_model = copy.deepcopy(model)
ref_opt = torch.optim.SGD(ref_model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Compiled (NKIPy) with dynamic shapes
comp_model = copy.deepcopy(model).to("nkipy")
comp_opt = torch.optim.SGD(comp_model.parameters(), lr=0.01, foreach=False)


@torch.compile(backend="nkipy", fullgraph=True, dynamic=True)
def forward_with_loss(m, x, target):
    out = m(x)
    loss = loss_fn(out, target)
    return out, loss


compiled_opt_step = torch.compile(comp_opt.step, backend="nkipy")

# Train with varying batch sizes
batch_sizes = [4, 8, 4, 16, 8]
for step, batch_size in enumerate(batch_sizes):
    torch.manual_seed(step)
    x = torch.randn(batch_size, 32)
    target = torch.randn(batch_size, 16)

    # Mark batch dimension as dynamic
    torch._dynamo.maybe_mark_dynamic(x, 0)
    torch._dynamo.maybe_mark_dynamic(target, 0)

    # Eager
    ref_opt.zero_grad()
    ref_loss = loss_fn(ref_model(x), target)
    ref_loss.backward()
    ref_opt.step()

    # Compiled
    comp_opt.zero_grad()
    _, comp_loss = forward_with_loss(comp_model, x.to("nkipy"), target.to("nkipy"))
    comp_loss.backward()
    compiled_opt_step()

    print(
        f"Step {step} (batch={batch_size}): "
        f"eager_loss={ref_loss.item():.4f}  "
        f"compiled_loss={comp_loss.cpu().item():.4f}"
    )

print("Training complete!")
