# VARIABLES
CUDAROOT   := /usr/local/cuda-5.5
CUDA_INC   := $(CUDAROOT)/targets/x86_64-linux/include
EIGEN_INC  := /usr/include/eigen3
SIGAR_INC  := /usr/include/sigar

CXX  := g++
NVCC := nvcc

DEBUG := 0

executable := drainage

src_root := src
obj_root := build/drainage

src_dirs := $(sort $(dir $(shell find $(src_root) -name '*.cpp' -o -name '*.cu')))

src_cpp := $(shell find $(src_root) -name '*.cpp')
src_cu  := $(shell find $(src_root) -name '*.cu')

obj_cpp := $(subst .cpp,.cpp.o,$(src_cpp))
obj_cu  := $(subst .cu,.cu.o,$(src_cu))

obj_cpp := $(patsubst $(src_root)/%,$(obj_root)/%,$(obj_cpp))
obj_cu  := $(patsubst $(src_root)/%,$(obj_root)/%,$(obj_cu))

INCS := -I$(CUDA_INC) -I$(SIGAR_INC) -I$(EIGEN_INC)
INCS += $(addprefix -I,$(src_dirs))

LDLIBS    := -lsigar -lcuda -lfftw3 -lgomp -lboost_program_options -lboost_filesystem -lboost_system

CXXFLAGS := -fopenmp
NVCCFLAGS := --compiler-bindir=$(CXX) -arch=sm_20

ifeq ($(DEBUG),1)
	CXXFLAGS += -O0 -DDEBUG -g
else
	CXXFLAGS += -O3
endif

.PHONY: all clean

all: dest $(executable)

dest:
	$(shell mkdir -p $(patsubst $(src_root)/%,$(obj_root)/%,$(src_dirs)))

$(executable): $(obj_cpp) $(obj_cu)
	$(NVCC) $^ $(LDLIBS) -o $@

$(obj_root)/%.cu.o: $(src_root)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCS) -c $< -o $@

$(obj_root)/%.cpp.o: $(src_root)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -c $< -o $@

clean:
	rm -rf $(obj_root) $(executable)

