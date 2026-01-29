import torch
import torch.distributed as dist
import torch.nn as nn

from torch_to_nkipy import init_nkipy_backend

torch.manual_seed(0)
torch.distributed.init_process_group("nkipy")

# Auto-detects rank/world_size from dist.get_rank()/dist.get_world_size()
init_nkipy_backend()


@torch.compile(backend="nkipy", fullgraph=True, dynamic=False)
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        h0 = self.gate_proj(x)
        h1 = self.up_proj(x)
        h2 = self.act_fn(h0) * h1
        h3 = self.down_proj(h2)
        h4 = dist.all_reduce(h3, op=dist.ReduceOp.SUM)
        return h4


model = MLP().to("nkipy")
input_tensor = torch.randn(2, 4).to("nkipy")

with torch.no_grad():
    out_tensor = model(input_tensor)
if dist.get_rank() == 0:
    print(f"out={out_tensor.cpu()}")

dist.destroy_process_group()
