#ifndef SMC
#define SMC
#include"device_function.cuh"
#include "mesh.h"
#include "gpu_info.h"

extern void smc_move( Phase *phase,GPU_info *gpu_info,MPI_info *mpi_info,int step);

extern void test_Phase_info_gpu(int cpu_index,int gpu_index,Phase *phase,GPU_info *gpu_info,MPI_info *mpi_info);

extern void test_stream(GPU_info *gpu_info);
#endif
