# Makefile for Relay P2P Engine pybind11 project (EFA RDMA read/write)

EFA_HOME ?= /opt/amazon/efa
NRT_HOME ?= /opt/aws/neuron
BUILD_DIR := ../build

# Compiler and flags
CXX := g++
CXXFLAGS := -O3 -shared -std=c++17 -fPIC -I. -Iinclude \
	-DRELAY_USE_EFA \
	-Wno-pointer-arith -Wno-sign-compare -Wno-unused-variable \
	-Wl,-rpath=/usr/local/lib -Wl,-rpath=/usr/lib/x86_64-linux-gnu

INC = -I./ -I${EFA_HOME}/include

# Try to use specific libglog.so.0 if available, otherwise use system libglog
LIBGLOG := $(shell test -f /usr/local/lib/libglog.so.0 && echo "/usr/local/lib/libglog.so.0" || echo "-lglog")
LIBS = -libverbs -lpthread $(LIBGLOG) -lgflags -L ${EFA_HOME}/lib -lefa

# Detect if NRT is available (for Neuron/Trainium support)
NRT_AVAILABLE := $(shell test -d $(NRT_HOME)/include/nrt && echo yes)
ifeq ($(NRT_AVAILABLE),yes)
    $(info NRT detected - enabling Neuron/Trainium support)
    CXXFLAGS += -DRELAY_ENABLE_NRT -Wl,-rpath=$(NRT_HOME)/lib
    INC += -I${NRT_HOME}/include
    LIBS += -L${NRT_HOME}/lib -lnrt
else
    $(info NRT not found - building without Neuron/Trainium support)
endif

# Python and pybind11 configuration
PYTHON          ?= python3
PYEXT           := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_LDFLAGS := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LDFLAGS') or '')")

# Target and source files
RELAY_EXT := _relay$(PYEXT)
SOURCES := engine.cc engine_pybind.cc
OBJECTS := $(addprefix $(BUILD_DIR)/,$(SOURCES:.cc=.o))

# Default target
all: $(BUILD_DIR)/$(RELAY_EXT)

$(BUILD_DIR)/$(RELAY_EXT): $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(OBJECTS) \
	      -o $@ $(PYTHON_LDFLAGS) $(CXXFLAGS) $(INC) $(LIBS)

# Compile source files
$(BUILD_DIR)/%.o: %.cc | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) $(INC) -c $< -o $@

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Test the module
test:
	$(MAKE) all
	$(PYTHON) tests/test_engine_write.py

# Install pybind11 if not available
install-deps:
	pip3 install pybind11

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the pybind11 module"
	@echo "  clean        - Remove build artifacts"
	@echo "  test         - Run the test script"
	@echo "  install-deps - Install pybind11 dependency"
	@echo "  help         - Show this help message"

.PHONY: all clean test install-deps help
