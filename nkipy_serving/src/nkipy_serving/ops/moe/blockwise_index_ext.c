#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <stdlib.h>

enum {
  SKIP_DMA = -1,
  SKIP_BLOCK = -2,
};

static PyObject *get_blockwise_expert_and_token_mapping_impl(PyObject *self,
                                                             PyObject *args,
                                                             PyObject *kwargs) {
  static char *kwlist[] = {
      "top_k_indices", "num_blocks",        "block_size",
      "num_experts",   "num_static_blocks", NULL,
  };

  PyObject *top_k_indices_obj = NULL;
  long long num_blocks_ll = 0;
  long long block_size_ll = 0;
  long long num_experts_ll = 0;
  long long num_static_blocks_ll = 0;

  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OLLLL", kwlist, &top_k_indices_obj, &num_blocks_ll,
          &block_size_ll, &num_experts_ll, &num_static_blocks_ll)) {
    return NULL;
  }

  if (num_blocks_ll < 0 || block_size_ll <= 0 || num_experts_ll <= 0 ||
      num_static_blocks_ll < 0) {
    PyErr_SetString(PyExc_RuntimeError, "invalid blockwise_index arguments");
    return NULL;
  }

  PyArrayObject *topk = (PyArrayObject *)PyArray_FROM_OTF(
      top_k_indices_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  if (topk == NULL) {
    return NULL;
  }
  if (PyArray_NDIM(topk) != 2) {
    Py_DECREF(topk);
    PyErr_SetString(PyExc_RuntimeError, "top_k_indices must be rank-2");
    return NULL;
  }

  const int64_t T = (int64_t)PyArray_DIM(topk, 0);
  const int64_t TOP_K = (int64_t)PyArray_DIM(topk, 1);
  const int64_t num_blocks = (int64_t)num_blocks_ll;
  const int64_t block_size = (int64_t)block_size_ll;
  const int64_t num_experts = (int64_t)num_experts_ll;
  const int64_t num_static_blocks = (int64_t)num_static_blocks_ll;
  const int32_t *topk_ptr = (const int32_t *)PyArray_DATA(topk);

  int64_t *tokens_per_expert =
      (int64_t *)calloc((size_t)num_experts, sizeof(int64_t));
  int64_t *blocks_per_expert =
      (int64_t *)calloc((size_t)num_experts, sizeof(int64_t));
  int64_t *start_block_per_expert =
      (int64_t *)calloc((size_t)num_experts, sizeof(int64_t));
  int64_t *current_token_per_expert =
      (int64_t *)calloc((size_t)num_experts, sizeof(int64_t));
  if (tokens_per_expert == NULL || blocks_per_expert == NULL ||
      start_block_per_expert == NULL || current_token_per_expert == NULL) {
    Py_DECREF(topk);
    free(tokens_per_expert);
    free(blocks_per_expert);
    free(start_block_per_expert);
    free(current_token_per_expert);
    return PyErr_NoMemory();
  }

  for (int64_t t = 0; t < T; ++t) {
    for (int64_t k = 0; k < TOP_K; ++k) {
      const int32_t e = topk_ptr[t * TOP_K + k];
      if (e >= 0) {
        if (e >= num_experts) {
          Py_DECREF(topk);
          free(tokens_per_expert);
          free(blocks_per_expert);
          free(start_block_per_expert);
          free(current_token_per_expert);
          PyErr_SetString(PyExc_RuntimeError, "expert id out of range");
          return NULL;
        }
        tokens_per_expert[e] += 1;
      }
    }
  }

  int64_t num_real_blocks = 0;
  for (int64_t e = 0; e < num_experts; ++e) {
    blocks_per_expert[e] = (tokens_per_expert[e] + block_size - 1) / block_size;
    start_block_per_expert[e] = num_real_blocks;
    num_real_blocks += blocks_per_expert[e];
  }

  if (num_real_blocks > num_blocks) {
    Py_DECREF(topk);
    free(tokens_per_expert);
    free(blocks_per_expert);
    free(start_block_per_expert);
    free(current_token_per_expert);
    PyErr_SetString(PyExc_RuntimeError, "num_real_blocks exceeds num_blocks");
    return NULL;
  }

  npy_intp block_to_expert_dims[1] = {(npy_intp)num_blocks};
  PyArrayObject *block_to_expert =
      (PyArrayObject *)PyArray_SimpleNew(1, block_to_expert_dims, NPY_INT8);
  if (block_to_expert == NULL) {
    Py_DECREF(topk);
    free(tokens_per_expert);
    free(blocks_per_expert);
    free(start_block_per_expert);
    free(current_token_per_expert);
    return NULL;
  }
  int8_t *block_to_expert_ptr = (int8_t *)PyArray_DATA(block_to_expert);
  for (int64_t i = 0; i < num_blocks; ++i) {
    block_to_expert_ptr[i] = (int8_t)SKIP_BLOCK;
  }
  int64_t cur_block = 0;
  for (int64_t e = 0; e < num_experts; ++e) {
    for (int64_t b = 0; b < blocks_per_expert[e]; ++b) {
      block_to_expert_ptr[cur_block++] = (int8_t)e;
    }
  }
  if (num_real_blocks > 1) {
    for (int64_t b = num_real_blocks - 1; b > 0; --b) {
      if (block_to_expert_ptr[b] == block_to_expert_ptr[b - 1] &&
          b != num_static_blocks) {
        block_to_expert_ptr[b] = (int8_t)SKIP_DMA;
      }
    }
  }

  npy_intp token_pos_dims[2] = {(npy_intp)num_blocks, (npy_intp)block_size};
  PyArrayObject *token_position_to_id =
      (PyArrayObject *)PyArray_SimpleNew(2, token_pos_dims, NPY_INT32);
  if (token_position_to_id == NULL) {
    Py_DECREF(topk);
    Py_DECREF(block_to_expert);
    free(tokens_per_expert);
    free(blocks_per_expert);
    free(start_block_per_expert);
    free(current_token_per_expert);
    return NULL;
  }
  int32_t *token_pos_ptr = (int32_t *)PyArray_DATA(token_position_to_id);
  for (int64_t i = 0; i < num_blocks * block_size; ++i) {
    token_pos_ptr[i] = (int32_t)SKIP_DMA;
  }

  for (int64_t t = 0; t < T; ++t) {
    for (int64_t k = 0; k < TOP_K; ++k) {
      const int32_t e = topk_ptr[t * TOP_K + k];
      if (e >= 0) {
        const int64_t global_pos = start_block_per_expert[e] * block_size +
                                   current_token_per_expert[e];
        const int64_t block_idx = global_pos / block_size;
        const int64_t within_block = global_pos % block_size;
        token_pos_ptr[block_idx * block_size + within_block] = (int32_t)t;
        current_token_per_expert[e] += 1;
      }
    }
  }

  Py_DECREF(topk);
  free(tokens_per_expert);
  free(blocks_per_expert);
  free(start_block_per_expert);
  free(current_token_per_expert);

  PyObject *out = PyTuple_New(3);
  if (out == NULL) {
    Py_DECREF(block_to_expert);
    Py_DECREF(token_position_to_id);
    return NULL;
  }
  PyTuple_SET_ITEM(out, 0, PyLong_FromLongLong((long long)num_real_blocks));
  PyTuple_SET_ITEM(out, 1, (PyObject *)block_to_expert);
  PyTuple_SET_ITEM(out, 2, (PyObject *)token_position_to_id);
  return out;
}

static PyMethodDef module_methods[] = {
    {
        "get_blockwise_expert_and_token_mapping",
        (PyCFunction)get_blockwise_expert_and_token_mapping_impl,
        METH_VARARGS | METH_KEYWORDS,
        "Build block_to_expert and token_position_to_id for blockwise MoE.",
    },
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "blockwise_index_ext",
    "Native blockwise MoE index helper.",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit_blockwise_index_ext(void) {
  import_array();
  return PyModule_Create(&module_def);
}
