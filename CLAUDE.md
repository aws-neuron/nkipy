# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

This is a NeuronPy implementation of model for AWS Neuron TRN2 instance. Key components:

- **DeviceKernel**: Core abstraction for compiling and executing kernels on Neuron devices (device_kernel.py)
- **DeviceTensor**: Wrapper for tensors that can run on Neuron hardware (device_tensor.py)
- **PrefillLayer**: Attention and MoE, where MoE implemented by blockwise processing (prefill_layer.py)
- **Model**: Model implementation (model.py)
- **kernels/**: Custom NeuronPy kernels for attention, feedforward, RMSNorm, RoPE, etc.
- **Config**: Centralized configuration with hardware-specific compiler flags (config.py)

## Commands

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_attention_prefill.py
```

## Kernel Development
- DO NOT END PROGRAM EARLY. WAIT TO THE END TO OBSERVE RESULT.

### Compilation Process
- To get build_dir, for `chat.sh`, it will be printed as `build_dir: xxx`. For unit test, it is `/tmp/build/test/worker{worker_id}`.
- Each DeviceKernel gets a subdirectory named as argument or function name if name is None
- Compiled kernel's input/output names may differ from code due to compiler bug. To check real names, use command `neuron-packager info ${NEFF_FILE}`, e.g. `neuron-packager info ${build_dir}/router/router.neff`
- After modifying kernel code or arguments, to trigger recompilation, remove the build cache folder
- Check compilation errors in `log-neuron-cc.txt` within the kernel's build cache directory

### Kernel Types
- **NKI kernels**: Custom kernels written in Neuron Kernel Interface (kernels/*_nki.py)
- **NeuronPy kernels**: Higher-level kernels using NeuronPy numpy-like API
- **Use NKI kernel inside NeuronPy kernel**: Use NKICustomOp. IMPORTANT: you should pass NKI kernel arguments as a list in NKICustomOp, as shown in @tests/test_lnc2.py#L63-77

### Collective
- collectives in kernel can automatically skip if group_size == 1. No need to specially handle

## Development Rules
- DO NOT TIMEOUT when running `chat.sh`

### Code Style
- IMPORTANT: keep code, test and comments succinct - only necessary ones
- Follow existing patterns in the codebase
- Try NOT add optional argument in a function to make user pass in explict argument, to reduce error

### Testing
- Place test files in `tests/` directory
- No need to call `np.random.seed()` or `torch.manual_seed()` in tests (conftest.py handles it)
- IMPORTANT: to test device kernel, you should use DeviceKernel instead of calling function directly with `is_neuronpy=True`. For an example follow @tests/test_blockwise_nki.py.
- Use `assert_allclose` in `utils.py` instead of `np.testing.assert_allclose`. IMPORTANT: use default tolerances unless requested by user. When there is numerical mismatch, try HARD to fix it. If you cannot fix, it is okay to leave the test fail.
- No need to print stats (e.g. mean, std, min, max), just `assert_allclose`.
- When using `assert_allclose`, use the default type instead of casting to fp32.
- If value is in bf16 numpy cpu, it can print properly (you will see error `ValueError: Unknown format code 'f' for object of type 'str'`). You should convert it into fp32 first and print