# integrate vLLM with NKIPy

## Goal

The goal of this project is to integrate vLLM with NKIPy for model serving on Trainium with NKIPy as the backend and vLLM as the frontend interacting with users for online serving.


## Code path

vLLM code path: /home/ubuntu/vllm-nkipy/vllm
nkipy code path: /home/ubuntu/vllm-nkipy/nkipy


## An example reference

vllm-neuron is a open-source project that integrates vLLM with nxdi, which is another backend for model serving on Trainium. The source code is in home/ubuntu/vllm-nkipy/private-vllm-neuron and the major integration logic can be found in vllm_neuron/backends/nxdi.


## How to achieve the goal

Step 1: understand how vllm-neuron integrates with vLLM, especially the workflow of how the model is initialized from vLLM APIs to vllm-neuron and how the vLLM endpoints are passed to vllm-neuron
Step 2: add a plugin in NKIPy to support vLLM integration with NKIPy as the backend (don't use any dependencies from vllm_neuron). You can first implement offline model serving at this stage and only need to enable model initialization and very basic request inference for TinyLlama.
Step 3: add unit tests and end-to-end tests to validate the correctness. Go back to Step 2 if end-to-end tests don't pass.
Step 4: enable model running on Neuron cores