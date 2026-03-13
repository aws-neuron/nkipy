#ifndef SPIKE_SRC_INCLUDE_NRT_WRAPPER_H
#define SPIKE_SRC_INCLUDE_NRT_WRAPPER_H

#include "error.h"

#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_profile.h>

#include <cstdint>
#include <string>

extern "C" {

// Underlying tensor and model declaration copied from NRT
// This is a temporary hack for implementing nonblocking/async operations,
// as I will need some underlying info that is not exposed
// Will not need these after explicit async is ready and stable from NRT

#define DX_CACHE_ALIGNED __attribute__((aligned(64)))

typedef enum nrt_tensor_mem_type {
  NRT_TENSOR_MEM_TYPE_INVALID = 0,
  NRT_TENSOR_MEM_TYPE_MALLOC,
  NRT_TENSOR_MEM_TYPE_DMA,
  NRT_TENSOR_MEM_TYPE_FAKE,
} nrt_tensor_mem_type_t;

// Memory, host or device that is used by
// a tensor.  The memory is ref counted and can be shared among
// multiple tensors.
typedef struct nrt_tensor_storage {
  uint32_t hbm_idx;
  size_t allocated_size;
  nrt_tensor_mem_type_t type;
  union {
    void *dmem; // dmem associated with addr, for tensor type
                // NRT_TENSOR_MEM_TYPE_DMA
    uint8_t
        *vmem; // malloc'ed memory for tensor type NRT_TENSOR_MEM_TYPE_MALLOC
  };
  volatile uint64_t ref_count DX_CACHE_ALIGNED;
  bool mem_owned_by_tensor;

  pthread_mutex_t tensor_op_cv_lock; // Lock for async exec. Used with
                                     // `tensor_op_cv` to block the thread while
                                     // there are still pending execs. If this
                                     // is NULL we are not in async exec mode.
  pthread_cond_t tensor_op_cv;       // used to block tensor op vars
  volatile uint64_t pending_exec_count_read
      DX_CACHE_ALIGNED; // count of pending execs that reads this location
  volatile uint64_t pending_exec_count_write
      DX_CACHE_ALIGNED; // count of pending execs that writes to this location
  int32_t vtpb_idx;     // same as vcore->vtpb_idx but -1 if no vcore for tensor
                        // (used for trace api)
} nrt_tensor_storage_t;

typedef struct nrt_tensor {
  char *name;                // optional name
  nrt_tensor_storage_t *sto; // the actual memory represented by the tensor
  // don't access directly, use helper functions to ensure correctness
  // params below allow a tensor to represent a slice of the memory
  // pointed by "sto"
  size_t _offset; // offset within the storage
  size_t _size;   // tensor size
  void *extra;    // used to store any metadata needed by the runtime

  volatile uint64_t ref_count
      DX_CACHE_ALIGNED; // refcount for tensor. Only when this is 0 can we free
                        // the tensor it is incremented by
                        // `tensor_get_reference` and decremented by
                        // `tensor_free`. Tensor will automatically be freed in
                        // `tensor_free` once ref_count is zero.
  volatile uint64_t output_completion_count
      DX_CACHE_ALIGNED; // used to track the completion count of an output
                        // tensor. 0 means not complete; 1 and above means the
                        // number of completions
} nrt_tensor_t;

typedef struct H_NN {
  uint32_t id;
} H_NN;

struct nrt_model {
  uint32_t start_vnc;      // VirtualNeuronCore start index
  uint32_t vnc_count;      // number of VirtualNeuronCore(s) requested
  uint32_t instance_index; // instance index which will execute on the next call
                           // to nrt_execute
  uint32_t instance_count; // number of loaded instances
  uint32_t gid;            // global id, for debug
  char name[256];
  H_NN h_nn[]; // kmgr model id (instance_count entries)
};
}

namespace spike {

// RAII wrapper for NRT runtime
class NrtRuntime {
public:
  NrtRuntime(const std::string &fw_version = "spike",
             const std::string &fal_version = "0.1.0", bool owned = true);
  ~NrtRuntime();

  // Non-copyable, movable
  NrtRuntime(const NrtRuntime &) = delete;
  NrtRuntime &operator=(const NrtRuntime &) = delete;
  NrtRuntime(NrtRuntime &&) = default;
  NrtRuntime &operator=(NrtRuntime &&) = default;

  static uint32_t get_visible_nc_count();
  static uint32_t get_total_nc_count();

private:
  bool initialized_;
  bool owned_;
};

} // namespace spike

#endif // SPIKE_SRC_INCLUDE_NRT_WRAPPER_H
