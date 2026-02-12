# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training tests for torch-to-nkipy.

Validates that forward + backward + optimizer step through the NKIPy backend
produces the same losses and weights as eager CPU execution.
"""

import copy

import pytest
import torch
import torch.nn as nn
from base import NKIPyTestBase

try:
    from nkipy.runtime import is_neuron_compatible

    NEURON_AVAILABLE = is_neuron_compatible()
except ImportError:
    NEURON_AVAILABLE = False

_NUM_STEPS = 3
_LR = 0.01
_RTOL = 1e-2
_ATOL = 1e-2


def _check_training(model, make_batch, num_steps=_NUM_STEPS, lr=_LR, rtol=_RTOL, atol=_ATOL):
    """Train eager vs compiled for *num_steps*, assert losses and weights match."""
    torch.manual_seed(42)

    ref_model = copy.deepcopy(model)
    comp_model = copy.deepcopy(model).to("nkipy")

    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=lr)
    comp_opt = torch.optim.SGD(comp_model.parameters(), lr=lr, foreach=False)

    loss_fn = nn.MSELoss()

    @torch.compile(backend="nkipy", fullgraph=True)
    def forward_with_loss(m, x, target):
        out = m(x)
        loss = loss_fn(out, target)
        return out, loss

    compiled_opt_step = torch.compile(comp_opt.step, backend="nkipy")

    ref_losses = []
    comp_losses = []

    for step in range(num_steps):
        torch.manual_seed(step)
        x, target = make_batch()

        # Eager
        ref_opt.zero_grad()
        ref_out = ref_model(x)
        ref_loss = loss_fn(ref_out, target)
        ref_loss.backward()
        ref_opt.step()
        ref_losses.append(ref_loss.item())

        # Compiled
        comp_opt.zero_grad()
        comp_out, comp_loss = forward_with_loss(comp_model, x.to("nkipy"), target.to("nkipy"))
        comp_loss.backward()
        compiled_opt_step()
        comp_losses.append(comp_loss.cpu().item())

    # Per-step losses
    for step, (rl, cl) in enumerate(zip(ref_losses, comp_losses)):
        torch.testing.assert_close(
            torch.tensor(cl), torch.tensor(rl), rtol=rtol, atol=atol,
            msg=f"Step {step} loss mismatch",
        )

    # Final weights
    for (pname, rp), (_, cp) in zip(
        ref_model.named_parameters(), comp_model.named_parameters()
    ):
        torch.testing.assert_close(
            cp.detach().cpu(), rp.detach(), rtol=rtol, atol=atol,
            msg=f"Param '{pname}' mismatch after {num_steps} steps",
        )


@pytest.mark.skipif(not NEURON_AVAILABLE, reason="Requires Neuron hardware")
class TestTraining(NKIPyTestBase):
    """Training correctness: eager CPU vs compiled NKIPy."""

    def test_training_linear(self):
        _check_training(
            nn.Linear(32, 16, bias=True),
            lambda: (torch.randn(4, 32), torch.randn(4, 16)),
        )

    def test_training_mlp(self):
        _check_training(
            nn.Sequential(
                nn.Linear(32, 64, bias=False), nn.ReLU(),
                nn.Linear(64, 64, bias=False), nn.ReLU(),
                nn.Linear(64, 16, bias=False),
            ),
            lambda: (torch.randn(4, 32), torch.randn(4, 16)),
        )
