#pragma once
#include "engine.h"

inline ConnID uccl_connect(RDMAEndPoint const& s, int remote_gpuidx,
                           std::string remote_ip, uint16_t remote_port) {
  return s->uccl_connect(remote_gpuidx, remote_ip, remote_port);
}
inline uint16_t get_p2p_listen_port(RDMAEndPoint const& s) {
  return s->get_p2p_listen_port();
}

inline ConnID uccl_accept(RDMAEndPoint const& s, std::string& remote_ip,
                          int* remote_gpuidx) {
  return s->uccl_accept(remote_ip, remote_gpuidx);
}

inline void stop_accept(RDMAEndPoint const& s) { s->stop_accept(); }

inline bool uccl_regmr(RDMAEndPoint const& s, void* data, size_t len,
                       struct P2PMhandle* mhandle) {
  return s->uccl_regmr(data, len, mhandle->mr_array) >= 0;
}

inline bool uccl_poll_ureq_once(RDMAEndPoint const& s,
                                struct ucclRequest* ureq) {
  if (ureq->type == ReqType::ReqTx || ureq->type == ReqType::ReqWrite) {
    s->sendRoutine();
    return s->checkSendComplete_once(ureq->n, ureq->engine_idx);
  } else if (ureq->type == ReqType::ReqRx) {
    s->recvRoutine();
    return s->checkRecvComplete_once(ureq->n, ureq->engine_idx);
  }
  LOG(ERROR) << "Invalid request type: " << ureq->type;
  return false;
}

inline int uccl_write_async(RDMAEndPoint const& s, Conn* conn,
                            P2PMhandle* local_mh, void* src, size_t size,
                            FifoItem const& slot_item, ucclRequest* ureq) {
  ureq->type = ReqType::ReqWrite;

  // Create RemoteMemInfo from FifoItem
  auto remote_mem = std::make_shared<RemoteMemInfo>();
  remote_mem->addr = slot_item.addr;
  remote_mem->length = slot_item.size;
  remote_mem->type = MemoryType::GPU;
  remote_mem->rkey_array.copyFrom(slot_item.padding);
  // Create RegMemBlock for local memory
  auto local_mem = std::make_shared<RegMemBlock>(src, size, MemoryType::GPU);
  local_mem->mr_array = local_mh->mr_array;

  auto req = std::make_shared<RDMASendRequest>(local_mem, remote_mem);
  req->to_rank_id = conn->uccl_conn_id_.flow_id;
  req->send_type = SendType::Write;

  ureq->engine_idx = s->writeOrRead(req);
  ureq->n = conn->uccl_conn_id_.flow_id;

  return ureq->engine_idx;
}

inline int prepare_fifo_metadata(RDMAEndPoint const& s, Conn* conn,
                                 P2PMhandle* mhandle, void const* data,
                                 size_t size, char* out_buf) {
  FifoItem remote_mem_info;
  remote_mem_info.addr = reinterpret_cast<uint64_t>(data);
  remote_mem_info.size = size;

  copyRKeysFromMRArrayToBytes(mhandle->mr_array,
                              static_cast<char*>(remote_mem_info.padding),
                              sizeof(remote_mem_info.padding));
  auto* rkeys1 = const_cast<RKeyArray*>(
      reinterpret_cast<RKeyArray const*>(remote_mem_info.padding));
  serialize_fifo_item(remote_mem_info, out_buf);
  return 0;
}

inline void uccl_deregmr(RDMAEndPoint const& s, P2PMhandle* mhandle) {
  s->uccl_deregmr(mhandle->mr_array);
}

inline bool initialize_rdma_ctx_for_gpu(RDMAEndPoint const& s, int dev) {
  return s->initialize_rdma_ctx_for_gpu(dev);
}
