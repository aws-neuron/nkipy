#include "engine.h"
#ifdef RELAY_ENABLE_NRT
#include "nrt_tensor.h"
#endif
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace {
struct InsidePythonGuard {
  InsidePythonGuard() { inside_python = true; }
  ~InsidePythonGuard() { inside_python = false; }
};

struct XferDesc {
  void const* addr;
  size_t size;
  uint64_t mr_id;
  std::vector<uint32_t> lkeys;
  std::vector<uint32_t> rkeys;
};

std::vector<uint8_t> serialize_xfer_descs(
    std::vector<XferDesc> const& xfer_desc_v) {
  size_t total_size = sizeof(size_t);
  for (auto const& desc : xfer_desc_v) {
    assert(desc.lkeys.size() == desc.rkeys.size());
    total_size += sizeof(uint64_t) + sizeof(size_t) + sizeof(size_t) +
                  desc.lkeys.size() * sizeof(uint32_t) * 2;
  }

  std::vector<uint8_t> result(total_size);
  uint8_t* p = result.data();
  auto emit = [&p](void const* src, size_t n) {
    std::memcpy(p, src, n);
    p += n;
  };

  size_t num_descs = xfer_desc_v.size();
  emit(&num_descs, sizeof(size_t));
  for (auto const& desc : xfer_desc_v) {
    uint64_t addr = reinterpret_cast<uint64_t>(desc.addr);
    size_t keys_count = desc.lkeys.size();
    emit(&addr, sizeof(uint64_t));
    emit(&desc.size, sizeof(size_t));
    emit(&keys_count, sizeof(size_t));
    emit(desc.lkeys.data(), keys_count * sizeof(uint32_t));
    emit(desc.rkeys.data(), keys_count * sizeof(uint32_t));
  }
  return result;
}

std::vector<XferDesc> deserialize_xfer_descs(
    std::vector<uint8_t> const& serialized_data) {
  if (serialized_data.empty()) return {};

  uint8_t const* p = serialized_data.data();
  uint8_t const* end = p + serialized_data.size();
  auto consume = [&p, end](void* dst, size_t n) {
    if (p + n > end)
      throw std::runtime_error("Invalid serialized XferDesc data");
    std::memcpy(dst, p, n);
    p += n;
  };

  size_t num_descs;
  consume(&num_descs, sizeof(size_t));

  std::vector<XferDesc> xfer_desc_v;
  xfer_desc_v.reserve(num_descs);
  for (size_t i = 0; i < num_descs; ++i) {
    XferDesc desc;
    uint64_t addr;
    size_t keys_count;
    consume(&addr, sizeof(uint64_t));
    desc.addr = reinterpret_cast<void const*>(addr);
    consume(&desc.size, sizeof(size_t));
    consume(&keys_count, sizeof(size_t));
    desc.lkeys.resize(keys_count);
    consume(desc.lkeys.data(), keys_count * sizeof(uint32_t));
    desc.rkeys.resize(keys_count);
    consume(desc.rkeys.data(), keys_count * sizeof(uint32_t));
    xfer_desc_v.push_back(std::move(desc));
  }
  return xfer_desc_v;
}

// Helper function to extract pointer and size from PyTorch tensor
std::pair<void const*, size_t> get_tensor_info(py::object tensor_obj) {
  // Get data pointer using Python's data_ptr() method
  if (!py::hasattr(tensor_obj, "data_ptr")) {
    throw std::runtime_error("Tensor does not have data_ptr() method");
  }
  py::object data_ptr_result = tensor_obj.attr("data_ptr")();
  uint64_t ptr = py::cast<uint64_t>(data_ptr_result);

  // Get number of elements
  if (!py::hasattr(tensor_obj, "numel")) {
    throw std::runtime_error("Tensor does not have numel() method");
  }
  py::object numel_result = tensor_obj.attr("numel")();
  int64_t numel = py::cast<int64_t>(numel_result);

  // Get element size
  if (!py::hasattr(tensor_obj, "element_size")) {
    throw std::runtime_error("Tensor does not have element_size() method");
  }
  py::object element_size_result = tensor_obj.attr("element_size")();
  int64_t element_size = py::cast<int64_t>(element_size_result);

  size_t total_size = static_cast<size_t>(numel * element_size);
  return std::make_pair(reinterpret_cast<void const*>(ptr), total_size);
}

#ifdef RELAY_ENABLE_NRT
/**
 * Python wrapper class for NRT tensor
 */
class NRTTensor {
 private:
  nrt_tensor_t* tensor_;
  void* address_;
  size_t size_;

 public:
  NRTTensor(int nc_idx, size_t size_bytes, char const* name = nullptr)
      : tensor_(nullptr), address_(nullptr), size_(0) {
    int status = nrt_util_allocate_device_tensor(nc_idx, size_bytes, &address_,
                                                 &tensor_);

    if (status != 0) {
      throw std::runtime_error("Failed to allocate NRT tensor: " +
                               std::to_string(status));
    }

    size_ = size_bytes;
  }

  ~NRTTensor() {
    if (tensor_) {
      nrt_util_free_tensor(&tensor_);
    }
  }

  // Prevent copying
  NRTTensor(NRTTensor const&) = delete;
  NRTTensor& operator=(NRTTensor const&) = delete;

  // Get the virtual address as an integer (for Python)
  uintptr_t get_address() const {
    return reinterpret_cast<uintptr_t>(address_);
  }

  // Get the size in bytes
  size_t get_size() const { return size_; }

  // Get the tensor pointer as an integer (for registration with Relay)
  uintptr_t get_va() const {
    if (!tensor_) return 0;
    return reinterpret_cast<uintptr_t>(nrt_tensor_get_va(tensor_));
  }

  // Create a PyTorch tensor backed by this NRT memory
  // Takes self parameter to attach reference and keep NRTTensor alive
  static py::object to_torch_impl(py::object self) {
    // Extract the C++ object
    NRTTensor& cpp_obj = self.cast<NRTTensor&>();

    // Import torch and ctypes modules
    py::module_ torch = py::module_::import("torch");
    py::module_ ctypes = py::module_::import("ctypes");

    // Create ctypes array type: (c_float * num_elements)
    py::object c_float = ctypes.attr("c_float");
    size_t num_elements = cpp_obj.size_ / sizeof(float);
    py::object array_type = c_float.attr("__mul__")(num_elements);

    // Create ctypes array from address: array_type.from_address(address)
    py::object ctypes_array = array_type.attr("from_address")(
        reinterpret_cast<uintptr_t>(cpp_obj.address_));

    // Create torch tensor using frombuffer (zero-copy)
    py::object frombuffer = torch.attr("frombuffer");
    py::object dtype = torch.attr("float32");
    py::object tensor = frombuffer(ctypes_array, py::arg("dtype") = dtype);

    // Attach the NRTTensor object to the tensor to keep it alive
    // This prevents the NRTTensor from being destroyed while the tensor is in
    // use
    tensor.attr("_nrt_owner") = self;

    return tensor;
  }
};
#endif  // RELAY_ENABLE_NRT

}  // namespace

PYBIND11_MODULE(_relay, m) {
  m.doc() = "Relay Engine - High-performance RDMA-based peer-to-peer transport";

  m.def("get_oob_ip", &relay::get_oob_ip, "Get the OOB IP address");

  // Feature flags
#ifdef RELAY_ENABLE_NRT
  m.attr("HAS_NRT_SUPPORT") = true;
#else
  m.attr("HAS_NRT_SUPPORT") = false;
#endif

  // Register XferDesc type
  py::class_<XferDesc>(m, "XferDesc")
      .def(py::init<>())
      .def_property(
          "addr",
          [](XferDesc const& d) { return reinterpret_cast<uint64_t>(d.addr); },
          [](XferDesc& d, uint64_t v) {
            d.addr = reinterpret_cast<void const*>(v);
          })
      .def_readwrite("size", &XferDesc::size)
      .def_readwrite("mr_id", &XferDesc::mr_id)
      .def_readwrite("lkeys", &XferDesc::lkeys)
      .def_readwrite("rkeys", &XferDesc::rkeys)
      .def("__repr__", [](XferDesc const& d) {
        return "<XferDesc addr=" +
               std::to_string(reinterpret_cast<uint64_t>(d.addr)) +
               " size=" + std::to_string(d.size) +
               " mr_id=" + std::to_string(d.mr_id) + ">";
      });

  // Endpoint class binding
  py::class_<Endpoint>(m, "Endpoint")
      .def(py::init([](uint32_t local_gpu_idx) {
             py::gil_scoped_release release;
             InsidePythonGuard guard;
             return std::make_unique<Endpoint>(local_gpu_idx);
           }),
           py::arg("local_gpu_idx"))
      .def(
          "start_passive_accept",
          [](Endpoint& self) { return self.start_passive_accept(); },
          "Start a background thread for accepting.")
      .def(
          "connect",
          [](Endpoint& self, std::string const& remote_ip_addr,
             int remote_gpu_idx, int remote_port) {
            uint64_t conn_id;
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.connect(remote_ip_addr, remote_gpu_idx,
                                     remote_port, conn_id);
            }
            return py::make_tuple(success, conn_id);
          },
          "Connect to a remote server", py::arg("remote_ip_addr"),
          py::arg("remote_gpu_idx"), py::arg("remote_port") = -1)
      .def(
          "add_remote_endpoint",
          [](Endpoint& self, py::bytes metadata_bytes) {
            uint64_t conn_id;
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              std::string buf = metadata_bytes;
              std::vector<uint8_t> metadata(buf.begin(), buf.end());
              success = self.add_remote_endpoint(metadata, conn_id);
            }
            return py::make_tuple(success, conn_id);
          },
          "Add remote endpoint - connect only once per remote endpoint.",
          py::arg("metadata_bytes"))
      .def(
          "get_metadata",
          [](Endpoint& self) {
            std::vector<uint8_t> metadata = self.get_metadata();
            return py::bytes(reinterpret_cast<char const*>(metadata.data()),
                             metadata.size());
          },
          "Return endpoint metadata as a list of bytes")
      .def_static(
          "parse_metadata",
          [](py::bytes metadata_bytes) {
            std::string buf = metadata_bytes;
            std::vector<uint8_t> metadata(buf.begin(), buf.end());
            auto [ip, port, gpu_idx] = Endpoint::parse_metadata(metadata);
            return py::make_tuple(ip, port, gpu_idx);
          },
          "Parse endpoint metadata to extract IP address, port, and GPU index",
          py::arg("metadata"))
      .def(
          "accept",
          [](Endpoint& self) {
            bool success;
            std::string remote_ip_addr;
            int remote_gpu_idx;
            uint64_t conn_id;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.accept(remote_ip_addr, remote_gpu_idx, conn_id);
            }
            return py::make_tuple(success, remote_ip_addr, remote_gpu_idx,
                                  conn_id);
          },
          "Accept an incoming connection")
      .def(
          "reg",
          [](Endpoint& self, uint64_t ptr, size_t size) {
            uint64_t mr_id;
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success =
                  self.reg(reinterpret_cast<void const*>(ptr), size, mr_id);
            }
            return py::make_tuple(success, mr_id);
          },
          "Register a data buffer", py::arg("ptr"), py::arg("size"))
      .def(
          "regv",
          [](Endpoint& self, std::vector<uintptr_t> const& ptrs,
             std::vector<size_t> const& sizes) {
            if (ptrs.size() != sizes.size())
              throw std::runtime_error("ptrs and sizes must match");

            std::vector<void const*> data_v;
            data_v.reserve(ptrs.size());
            for (auto p : ptrs)
              data_v.push_back(reinterpret_cast<void const*>(p));

            std::vector<uint64_t> mr_ids;
            bool ok;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              ok = self.regv(data_v, sizes, mr_ids);
            }
            return py::make_tuple(ok, py::cast(mr_ids));
          },
          py::arg("ptrs"), py::arg("sizes"),
          "Batch-register multiple memory regions and return [ok, mr_id_list]")
      .def(
          "register_memory",
          [](Endpoint& self, py::list tensor_list) {
            std::vector<void const*> ptrs;
            std::vector<size_t> sizes;
            size_t list_len = py::len(tensor_list);
            ptrs.reserve(list_len);
            sizes.reserve(list_len);

            for (size_t i = 0; i < list_len; ++i) {
              py::object tensor_obj = tensor_list[i];
              if (!py::hasattr(tensor_obj, "data_ptr")) {
                throw std::runtime_error(
                    "Object at index " + std::to_string(i) +
                    " is not a tensor (missing data_ptr() method)");
              }
              auto [ptr, size] = get_tensor_info(tensor_obj);
              ptrs.push_back(ptr);
              sizes.push_back(size);
            }

            std::vector<XferDesc> xfer_desc_v;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;

              std::vector<uint64_t> mr_id_v;
              if (!self.regv(ptrs, sizes, mr_id_v)) {
                return std::vector<XferDesc>{};
              }

              xfer_desc_v.reserve(list_len);
              for (size_t i = 0; i < list_len; i++) {
                auto mhandle = self.get_mhandle(mr_id_v[i]);
                assert(mhandle != nullptr);
                XferDesc xfer_desc;
                xfer_desc.addr = ptrs[i];
                xfer_desc.size = sizes[i];
                xfer_desc.mr_id = mr_id_v[i];
                for (size_t j = 0; j < kNICContextNumber; j++) {
                  auto mr = mhandle->mr_array.getKeyByContextID(j);
                  assert(mr != nullptr);
                  xfer_desc.lkeys.push_back(mr->lkey);
                  xfer_desc.rkeys.push_back(mr->rkey);
                }
                xfer_desc_v.push_back(std::move(xfer_desc));
              }
            }
            return xfer_desc_v;
          },
          "Register memory for a list of PyTorch tensors and return transfer "
          "descriptors",
          py::arg("tensor_list"))
      .def(
          "get_serialized_descs",
          [](Endpoint& /*self*/, py::list desc_list) {
            std::vector<XferDesc> xfer_desc_v;
            size_t list_len = py::len(desc_list);
            xfer_desc_v.reserve(list_len);
            for (size_t i = 0; i < list_len; ++i) {
              xfer_desc_v.push_back(py::cast<XferDesc const&>(desc_list[i]));
            }

            auto serialized = serialize_xfer_descs(xfer_desc_v);
            return py::bytes(reinterpret_cast<const char*>(serialized.data()),
                             serialized.size());
          },
          "Serialize transfer descriptors to bytes for network transmission",
          py::arg("desc_list"))
      .def(
          "deserialize_descs",
          [](Endpoint& /*self*/, py::bytes serialized_bytes) {
            std::string buf = serialized_bytes;
            std::vector<uint8_t> serialized_data(buf.begin(), buf.end());
            return deserialize_xfer_descs(serialized_data);
          },
          "Deserialize bytes to transfer descriptors",
          py::arg("serialized_bytes"))
      .def(
          "transfer",
          [](Endpoint& self, uint64_t conn_id, std::string const& op_name,
             py::list local_desc_list, py::list remote_desc_list) {
            size_t n = py::len(local_desc_list);
            if (n != py::len(remote_desc_list)) {
              throw std::runtime_error(
                  "Local and remote descriptors must have the same size");
            }
            if (op_name != "write") {
              throw std::runtime_error("Invalid op_name: " + op_name +
                                       " (only 'write' is supported)");
            }

            uint64_t transfer_id;
            bool success;

            if (n == 1) {
              auto const& ldesc = py::cast<XferDesc const&>(local_desc_list[0]);
              auto const& rdesc =
                  py::cast<XferDesc const&>(remote_desc_list[0]);

              FifoItem fifo_item;
              fifo_item.addr = reinterpret_cast<uint64_t>(rdesc.addr);
              fifo_item.size = rdesc.size;
              std::memcpy(fifo_item.padding, rdesc.rkeys.data(),
                          rdesc.rkeys.size() * sizeof(uint32_t));

              {
                py::gil_scoped_release release;
                InsidePythonGuard guard;
                success = self.write(conn_id, ldesc.mr_id,
                                     const_cast<void*>(ldesc.addr), ldesc.size,
                                     fifo_item);
              }
              transfer_id = 0;
            } else {
              std::vector<FifoItem> slot_item_v;
              std::vector<uint64_t> mr_id_v;
              std::vector<void*> src_v;
              std::vector<size_t> size_v;
              slot_item_v.reserve(n);
              mr_id_v.reserve(n);
              src_v.reserve(n);
              size_v.reserve(n);

              for (size_t i = 0; i < n; ++i) {
                auto const& ldesc =
                    py::cast<XferDesc const&>(local_desc_list[i]);
                auto const& rdesc =
                    py::cast<XferDesc const&>(remote_desc_list[i]);

                FifoItem fifo_item;
                fifo_item.addr = reinterpret_cast<uint64_t>(rdesc.addr);
                fifo_item.size = rdesc.size;
                std::memcpy(fifo_item.padding, rdesc.rkeys.data(),
                            rdesc.rkeys.size() * sizeof(uint32_t));

                slot_item_v.push_back(fifo_item);
                mr_id_v.push_back(ldesc.mr_id);
                src_v.push_back(const_cast<void*>(ldesc.addr));
                size_v.push_back(ldesc.size);
              }

              {
                py::gil_scoped_release release;
                InsidePythonGuard guard;
                success = self.writev(conn_id, mr_id_v, src_v, size_v,
                                      slot_item_v, n);
              }
              transfer_id = 0;
            }
            return py::make_tuple(success, transfer_id);
          },
          "Start a transfer and return (success, transfer_id)",
          py::arg("conn_id"), py::arg("op_name"), py::arg("local_desc_list"),
          py::arg("remote_desc_list"))
      .def(
          "dereg",
          [](Endpoint& self, uint64_t mr_id) {
            bool ok;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              ok = self.dereg(mr_id);
            }
            return ok;
          },
          "Deregister a memory region", py::arg("mr_id"))
      .def(
          "write",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id, uint64_t ptr,
             size_t size, py::bytes meta_blob) {
            std::string buf = meta_blob;
            if (buf.size() != sizeof(FifoItem))
              throw std::runtime_error(
                  "meta must be exactly 64 bytes (serialized FifoItem)");

            FifoItem item;
            deserialize_fifo_item(buf.data(), &item);
            bool success;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              success = self.write(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                                   size, item);
            }
            return success;
          },
          "RDMA-WRITE into a remote buffer using metadata from advertise(); "
          "`meta` is the 64-byte serialized FifoItem returned by the peer",
          py::arg("conn_id"), py::arg("mr_id"), py::arg("ptr"), py::arg("size"),
          py::arg("meta"))
      .def(
          "writev",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<uint64_t> ptr_v, std::vector<size_t> size_v,
             py::list meta_blob_v, size_t num_iovs) {
            if (mr_id_v.size() != num_iovs || ptr_v.size() != num_iovs ||
                size_v.size() != num_iovs || py::len(meta_blob_v) != num_iovs) {
              throw std::runtime_error(
                  "All input vectors/lists must have length num_iovs");
            }
            std::vector<FifoItem> item_v;
            item_v.reserve(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              std::string buf = py::cast<py::bytes>(meta_blob_v[i]);
              if (buf.size() != sizeof(FifoItem))
                throw std::runtime_error(
                    "meta must be exactly 64 bytes (serialized FifoItem)");
              FifoItem item;
              deserialize_fifo_item(buf.data(), &item);
              item_v.push_back(item);
            }
            std::vector<void*> data_v;
            data_v.reserve(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              data_v.push_back(reinterpret_cast<void*>(ptr_v[i]));
            }
            bool ok;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              ok = self.writev(conn_id, mr_id_v, data_v, size_v, item_v,
                               num_iovs);
            }
            return ok;
          },
          "RDMA-WRITE into multiple remote buffers using metadata from "
          "advertisev(); "
          "`meta_blob_v` is a list of 64-byte serialized FifoItem returned by "
          "the peer",
          py::arg("conn_id"), py::arg("mr_id_v"), py::arg("ptr_v"),
          py::arg("size_v"), py::arg("meta_blob_v"), py::arg("num_iovs"))
      .def(
          "advertise",
          [](Endpoint& self, uint64_t conn_id, uint64_t mr_id,
             uint64_t ptr,  // raw pointer passed from Python
             size_t size) {
            char serialized[sizeof(FifoItem)]{};  // 64-byte scratch buffer
            bool ok;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              ok = self.advertise(conn_id, mr_id, reinterpret_cast<void*>(ptr),
                                  size, serialized);
            }
            /* return (success, bytes) — empty bytes when failed */
            return py::make_tuple(
                ok, ok ? py::bytes(serialized, sizeof(FifoItem)) : py::bytes());
          },
          "Expose a registered buffer for the peer to RDMA-READ or RDMA-WRITE",
          py::arg("conn_id"), py::arg("mr_id"), py::arg("ptr"), py::arg("size"))
      .def(
          "advertisev",
          [](Endpoint& self, uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<uint64_t> ptr_v, std::vector<size_t> size_v,
             size_t num_iovs) {
            std::vector<char*> serialized_vec(num_iovs);
            for (size_t i = 0; i < num_iovs; ++i) {
              serialized_vec[i] = new char[sizeof(FifoItem)];
              memset(serialized_vec[i], 0, sizeof(FifoItem));
            }
            std::vector<void*> data_v;
            data_v.reserve(ptr_v.size());
            for (uint64_t ptr : ptr_v) {
              data_v.push_back(reinterpret_cast<void*>(ptr));
            }
            bool ok;
            {
              py::gil_scoped_release release;
              InsidePythonGuard guard;
              ok = self.advertisev(conn_id, mr_id_v, data_v, size_v,
                                   serialized_vec, num_iovs);
            }

            py::list py_bytes_list;
            for (size_t i = 0; i < num_iovs; ++i) {
              py_bytes_list.append(
                  py::bytes(serialized_vec[i], sizeof(FifoItem)));
            }
            for (size_t i = 0; i < num_iovs; ++i) {
              delete[] serialized_vec[i];
            }
            return py::make_tuple(ok, py_bytes_list);
          },
          "Expose multiple registered buffers for the peer to RDMA-READ or "
          "RDMA-WRITE",
          py::arg("conn_id"), py::arg("mr_id_v"), py::arg("ptr_v"),
          py::arg("size_v"), py::arg("num_iovs"))
      .def(
          "conn_id_of_rank", &Endpoint::conn_id_of_rank,
          "Get the connection ID for a given peer rank (or UINT64_MAX if none)",
          py::arg("rank"))
      .def("__repr__", [](Endpoint const& e) { return "<Relay Endpoint>"; });

#ifdef RELAY_ENABLE_NRT
  // NRT Tensor utilities (only available on Neuron/Trainium)
  py::class_<NRTTensor>(m, "NRTTensor")
      .def(py::init<int, size_t, char const*>(), py::arg("nc_idx"),
           py::arg("size_bytes"), py::arg("name") = nullptr,
           "Allocate an NRT tensor on device")
      .def("get_address", &NRTTensor::get_address,
           "Get the virtual address of the tensor (for frombuffer)")
      .def("get_size", &NRTTensor::get_size,
           "Get the size of the tensor in bytes")
      .def("get_va", &NRTTensor::get_va,
           "Get the virtual address for Relay registration")
      .def(
          "to_torch",
          [](py::object self) { return NRTTensor::to_torch_impl(self); },
          "Create a PyTorch tensor backed by this NRT memory (zero-copy, keeps "
          "NRTTensor alive)")
      .def("__repr__", [](NRTTensor const& t) {
        return "<NRTTensor address=" + std::to_string(t.get_address()) +
               " size=" + std::to_string(t.get_size()) + ">";
      });

  // Factory function: create NRT tensor and return PyTorch tensor directly
  m.def(
      "create_nrt_tensor",
      [](int nc_idx, size_t size_bytes, char const* name) {
        // Create NRTTensor as a Python object
        py::object nrt_obj = py::cast(new NRTTensor(nc_idx, size_bytes, name),
                                      py::return_value_policy::take_ownership);
        // Convert to torch and return (keeps NRTTensor alive via _nrt_owner)
        return NRTTensor::to_torch_impl(nrt_obj);
      },
      py::arg("nc_idx"), py::arg("size_bytes"), py::arg("name") = nullptr,
      "Create an NRT tensor and return as PyTorch tensor (NRT memory kept "
      "alive automatically)");

  m.def("nrt_init", &nrt_util_init, "Initialize NRT runtime (idempotent)");
#endif  // RELAY_ENABLE_NRT
}