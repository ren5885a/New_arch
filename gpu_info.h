#ifndef GPU_INFO_H
#define GPU_INFO_H

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "struct.h"
typedef struct {
	
	
	cudaDeviceProp prop[64];
	int GPU_N;
	int whichGPUs[20]; //maximal gpu number is set to 20

	cudaStream_t *stream;	

	int gridNx;
	int gridNy;
	int gridNz;

	std::vector<curandStatePhilox4_32_10_t *> state;

	int polymerNx;
	int polymerNy;
	int polymerNz;
	
}GPU_info;


extern void init_cuda(MPI_info *mpi_info,GPU_info *gpu_info,int display);

#endif
