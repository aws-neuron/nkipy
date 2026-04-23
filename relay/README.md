## Build

```bash
cd relay/src
make all PYTHON=/path/to/.venv/bin/python3
```

## Test on Trn

### 2 nodes x 1 rank/node
```
# master node
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=0 --master_addr=<MASTER_IP> benchmarks/benchmark_relay_write_neuron.py --local-nc-idx 5 --device <trn or cpu>

# worker node
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=<MASTER_IP> benchmarks/benchmark_relay_write_neuron.py --local-nc-idx 5 --device <trn or cpu>
```

### 1 node x 2 ranks/node
```
torchrun --nnodes=1 --nproc_per_node=2 benchmarks/benchmark_relay_write_neuron.py --local-nc-idx-group 5,10 --device <trn or cpu>
```
