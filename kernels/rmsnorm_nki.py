import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import math

@nki.compiler.skip_middle_end_transformations
@nki.jit(
    debug_kernel=True,
    show_compiler_tb=True,
)
def rmsnorm(
    x: nl.ndarray, 
    weight: nl.ndarray, 
    eps: float = 1e-6
) -> nl.ndarray:
    """
    Perform RMSNorm on input tensor using NKI.

    :param x: Tensor to perform RMSNorm on, which has shape [B, S, H].
    :param weight: Weight to apply on the rmsnorm, which has shape [1, H].
    :param eps: Small value to avoid division by zero
    :return: The HBM tensor with rmsnorm performed.
    """
    # Use float32 to reduce numerical error
    compute_dtype = nl.float32

    # Create output tensor with original shape and dtype
    output = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.hbm)
    
    if x.ndim == 2:
        B, H = x.shape
    elif x.ndim == 3:
        B, S, H = x.shape
        assert S == 1, "Only support tokengen"
    else:
        raise ValueError(f"Malformed shape of x {x.shape}")
    
    if weight.ndim == 2:
        assert weight.shape == (1, H), f"Malformed shape of weight {weight.shape}"
    elif weight.ndim == 1:
        assert weight.shape == (H,), f"Malformed shape of weight {weight.shape}"
    else:
        raise ValueError(f"Malformed shape of weight {weight.shape}")

    # unify shapes
    x = x.reshape((B, H))
    output_reshaped = output.reshape((B, H))
    weight = weight.reshape((H,))

    H0 = nl.tile_size.pmax # 128
    H1 = math.ceil(H / H0) # number of element in each partition

    result = nl.zeros((H0, B, H1), dtype=x.dtype, buffer=nl.sbuf)

    # Allocate input tensors
    input_sb = result[:, :, :] # Reuse result tensor
    weight_sb = nl.zeros((H0, H1), dtype=weight.dtype, buffer=nl.sbuf)


    # Load hidden (B, H) to (H0, B, H1)
    i_p, i_f0, i_f1 = nl.mgrid[0:H0, 0:B, 0:H1]
    mask = i_p * H1 + i_f1 < H
    nisa.dma_copy(src=x[i_f0, i_p * H1 + i_f1], dst=input_sb[i_p, i_f0, i_f1] , mask=mask)

    # Load weight (H) to (H0, H1)
    i_p, i_f1 = nl.mgrid[0:H0, 0:H1]
    mask = i_p * H1 + i_f1 < H
    nisa.dma_copy(src=weight[i_p * H1 + i_f1], dst=weight_sb[i_p, i_f1], mask=mask)
    
    # Compute x^2
    i_p, i_f0, i_f1 = nl.mgrid[0:H0, 0:B, 0:H1]
    z = nl.ndarray((H0, B, H1), dtype=compute_dtype, buffer=nl.sbuf)

    zero_bias = nl.ndarray((H0, 1), dtype=compute_dtype, buffer=nl.sbuf)
    zero_bias[...] = 0.0
    z[...] = nisa.activation(nl.square, input_sb, bias=zero_bias)

    # Sum across H dimension
    reduced = nl.ndarray((H0, B), dtype=compute_dtype, buffer=nl.sbuf)
    reduced[...] = nisa.tensor_reduce(nl.add, z, axis=(2))

    # Apply weight to input: x * weight
    input_sb[i_p, i_f0, i_f1] = nisa.tensor_tensor(input_sb[i_p, i_f0, i_f1], weight_sb[i_p, i_f1], nl.multiply)

    # Reduce across partitions
    rmsnorm_reduction_const = nisa.memset((H0, H0), value=1.0, dtype=compute_dtype)
    final_reduced = nl.ndarray((H0, B), dtype=nl.float32, buffer=nl.psum)
    final_reduced[...] = nisa.nc_matmul(rmsnorm_reduction_const, reduced)

    # Compute 1/sqrt(mean(x^2) + eps)
    eps_loaded = nisa.memset((H0, 1), value=eps, dtype=compute_dtype)
    rms_inv = nl.ndarray((H0, B), dtype=compute_dtype, buffer=nl.sbuf)
    rms_inv[...] = nisa.activation(nl.rsqrt, final_reduced, scale=(1.0/H), bias=eps_loaded)

    # Apply normalization: (x * weight) / sqrt(mean(x^2) + eps)
    result[:, :, :] = nisa.tensor_tensor(input_sb, rms_inv[i_p, i_f0], nl.multiply)

    
    # Store result back to HBM
    i_p, i_f0, i_f1 = nl.mgrid[0:H0, 0:B, 0:H1]
    mask = i_p * H1 + i_f1 < H
    nisa.dma_copy(src=result[i_p, i_f0, i_f1], dst=output_reshaped[i_f0, i_p * H1 + i_f1], mask=mask)

    return output
