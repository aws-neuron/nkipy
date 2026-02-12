# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple training example using the NKIPy backend.

Trains a small MLP for a few steps on Neuron hardware and compares
with eager CPU results.

Usage:
    cd nkipy/
    uv run python torch-to-nkipy/examples/basics/example_training.py
"""

import copy

import torch
import torch.nn as nn

from torch_to_nkipy.backend.nkipy_backend import init_nkipy_backend

torch.manual_seed(0)

init_nkipy_backend()

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

# Compiled (NKIPy)
comp_model = copy.deepcopy(model).to("nkipy")
comp_opt = torch.optim.SGD(comp_model.parameters(), lr=0.01, foreach=False)


@torch.compile(backend="nkipy", fullgraph=True)
def forward_with_loss(m, x, target):
    out = m(x)
    loss = loss_fn(out, target)
    return out, loss


compiled_opt_step = torch.compile(comp_opt.step, backend="nkipy")

# Train for 3 steps
for step in range(3):
    torch.manual_seed(step)
    x = torch.randn(4, 32)
    target = torch.randn(4, 16)

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

    print(f"Step {step}: eager_loss={ref_loss.item():.4f}  compiled_loss={comp_loss.cpu().item():.4f}")

print("Training complete!")
