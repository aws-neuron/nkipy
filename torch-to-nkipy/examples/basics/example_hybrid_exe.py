import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_to_nkipy import init_nkipy_backend

torch.manual_seed(0)

init_nkipy_backend()


@torch.compile(backend="nkipy", fullgraph=True, dynamic=False)
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        with torch.no_grad():
            self.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return F.linear(x, self.weight, None)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.fc2 = Linear(8, 4)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        h0 = self.fc1(x)
        h1 = h0.cpu()
        h2 = self.act_fn(h1)
        h3 = h2.to("nkipy")
        h4 = self.fc2(h3)
        return h4


model = MLP().to("nkipy")
input_tensor = torch.randn(2, 4).to("nkipy")
with torch.no_grad():
    out_tensor = model(input_tensor)
print(out_tensor.cpu())
