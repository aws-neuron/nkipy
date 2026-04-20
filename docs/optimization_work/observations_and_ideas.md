# Optimize endpoint latency of P2P weight transfer in NKIPy plugin


## System settings

There are two engines running, a server engine and a receiver engine, for P2P weight transfer.

Two major endpoints:
- /wake_up: the server engine will push the model weights from its neuron cores to the corresponding neuron cores in the receiver engine
- /sleep: the engine will release all the neuron resources it occupies, allowing other engines to occupy neuron resources for serving

## Goal

Optimize the latency of both /wake_up and /sleep endpoint in p2p weight transfer.
- /wake_up: the latency is measured as the time period between when the endpoint is sent out to it returns. When it returns, it indicates the engine is ready for model serving.
- /sleep: the latency is measured as the time period between when the endpoint is sent out to it returns. When it returns, it indicates the occupied neuron resources have been released and they are ready for another engine.


## A naive design

### /wake_up design 

All the operations are blocking, including MR registration, P2P transfer, and RDMA deregistration.

### /sleep design

All the operations are blocking. Since RDMA has been deregistrated, the overhead is dominiated by nrt_close() that cleans up tensors on neuron cores.

## An optimized design

In order to make the receiver engine ready for model serving, the only blocking operation is just P2P transfer to materialize the model weights on HBM.
We pre-register MRs at the server side to avoid the MR registration overhead during /wake_up. Similarly, model serving doesn't depend on RDMA deregistration. 
We don't deregister MRs at the server side and we make RDMA deregistration unblocking at that receiver side so that /wake_up endpoint returns right after P2P transfer completes.

### Tradeoff

Although the optimization reduces the /wake_up latency, it worsens the /sleep latency.
- On the server side: Because MRs are pre-registered, when the server receives a /sleep endpoint, it has to deregister MRs synchronously before it can release all the neuron resources. This RDMA deregistration will increase the /sleep latency to more than 20s from 2s.
- On the receiver side: We assume MRs are deregistered asynchronously after /wake_up and the expected latency is around 2s.

## An alternative design

In /wake_up, we only make MR registration and P2P transfer blocking. We deregister MRs at both server and receiver sides but make this operation unblocking.

- Pros: the /sleep endpoint won't be blocked by MR deregistration at the server side.
- Cons: it increases /wake_up latency by the MR registration overhead, which is around 2s.

## Task description

Please analyze the designs above and confirm the pros and cons. Choose one design or propose a better design that minimizes /wake_up and /sleep latency.
