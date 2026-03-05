Tutorial: https://quip-amazon.com/augoA8iOU7Ck/NeuronPy-gpt-oss-120b-Tutorials-082025

## vLLM NKIPy plugin

### Installation
Please install install Neuron Compiler and NKIPy first.

Install pytorch CPU
```bash
pip3 install torch=="2.7.0+cpu" --index-url https://download.pytorch.org/whl/cpu
```

Install vllm (v0.10.0)
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.10.0
VLLM_TARGET_DEVICE="empty" pip install -e .
```
TODO: newer version of vLLM requires plugin to implement more interfaces

Install vLLM NKIPy plugin
```bash
cd NeuronPyExps
pip install -e .
```

### Test plugin
```bash
PYTHONPATH=. VLLM_USE_V1=1 pytest tests_plugin/offline_inference/test_vllm_offline_integration.py -v --capture=tee-sys
```
