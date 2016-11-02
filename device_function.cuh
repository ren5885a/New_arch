#ifndef DEVICE_H
#define DEVICE_H

#include "struct.h"
#include "gpu_info.h"
#include <curand_kernel.h>

#define THREAD_PER_BLOCK 1024

__global__ void setup_curand(Phase_info_gpu *phase_info_gpu,int gpu_index,unsigned int seed,curandStatePhilox4_32_10_t *state);


__global__ void initialize_coord(float *pos,Phase_info_gpu *phase_info_gpu,Poly_arch *poly_arch,curandStatePhilox4_32_10_t *state);

static __device__ unsigned int cell_coordinate_to_index(Phase_info_gpu *phase_info_gpu, const int x, const int y, const int z);

__device__ unsigned int cell_coordinate_to_index(Phase_info_gpu *phase_info_gpu, const int x, const int y, const int z){
    //Unified data layout [type][x][y][z]
  return x + y*phase_info_gpu->nx + z*phase_info_gpu->nx*phase_info_gpu->ny ;
}


__global__ void reduce_field_double(double *density_dst,double *density_res,long size,int GPU_N);

__global__ void reduce_field_int(int *density_dst,int *density_res,long size,int GPU_N);

__global__ void omega_field_update(int *density,double *omega_field,Phase_info_gpu *phase_info_gpu);

__global__ void coord_to_density(float *pos,int *density,Phase_info_gpu *phase_info_gpu,Poly_arch *poly_arch);

__global__ void init_array(int *density,int size);

__global__ void mc_polymer_move_arr(Phase_info_gpu *phase_info_gpu,Poly_arch *poly_arch,float *pos,curandStatePhilox4_32_10_t *state,double *omega_field_unified);

__global__ void test_stream(double *pa);
#endif

