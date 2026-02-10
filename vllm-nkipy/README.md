# vllm-nkipy

A plugin for [vLLM](https://github.com/vllm-project/vllm) that enables model execution on AWS Neuron hardware using Dynamo2Neuron and NKIPy.

## Overview

vllm-nkipy provides a platform integration for vLLM that allows running large language models (LLMs) on AWS Trainium/Inferentia accelerators. It leverages:

- ~~[Dynamo2Neuron](https://github.com/aws-neuron/Dynamo2Neuron) to compile PyTorch models to Neuron~~
- ~~[NKIPy](https://github.com/aws/nkipy) for execution on Neuron devices~~

## Installation

Install vllm (v0.11.0) firstly.

```bash
VLLM_TARGET_DEVICE="empty" uv pip install -e .
```

Then, install vllm-nkipy

```bash
uv pip install -e .
```

## Supported Features

- Inference with large language models on AWS Neuron hardware
- Integration with vLLM's serving capabilities
- Native PyTorch attention implementation

## Supported Models

TODO

## Usage

TODO

```
cd examples/
VLLM_USE_V1=1 python offline_inference_neuron.py
```


## License

Apache 2.0
