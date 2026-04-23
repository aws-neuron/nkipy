# SPDX-License-Identifier: (GPL-2.0 OR BSD-2-Clause)

# !!! Remember to conda deactivate before running make !!!

CUDA_HOME:=/usr/local/cuda

COMMON_CXXFLAGS = -I /opt/amazon/efa/include -L /opt/amazon/efa/lib -lfabric -libverbs -lefa -lpthread \
	-I ../../../include -I ../.. -I ../../../collective/rdma -std=c++17 -lglog -lgflags

# CUDA flags (for GPU tests)
CUDA_CXXFLAGS = $(COMMON_CXXFLAGS) -I$(CUDA_HOME)/include -L $(CUDA_HOME)/lib64 -lcudart -lcuda

# Neuron/Trainium flags (NRT for HBM allocation + DMA-BUF)
NEURON_HOME:=/opt/aws/neuron
NEURON_CXXFLAGS = $(COMMON_CXXFLAGS) -I$(NEURON_HOME)/include -L$(NEURON_HOME)/lib -lnrt -Wl,-rpath,$(NEURON_HOME)/lib

HEADERS = ../define.h ../rdma_device.h ../rdma_context.h ../rdma_channel.h ../rdma_channel_group.h \
	../rdma_ctrl_channel.h ../rdma_endpoint.h ../memory_allocator.h ../epoll_client.h ../epoll_server.h

cuda: test_rmda_endpoint

neuron: test_rmda_endpoint_neuron

test_rmda_endpoint: test_rmda_endpoint.cpp $(HEADERS)
	g++ -O3 -g test_rmda_endpoint.cpp -o test_rmda_endpoint $(CUDA_CXXFLAGS)

test_rmda_endpoint_neuron: test_rmda_endpoint_neuron.cpp $(HEADERS)
	g++ -O3 -g test_rmda_endpoint_neuron.cpp -o test_rmda_endpoint_neuron $(NEURON_CXXFLAGS)

clean:
	rm -f test_rmda_endpoint test_rmda_endpoint_neuron *.o