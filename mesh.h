#ifndef MESH_GPU
#define MESH_GPU
#include "struct.h"

#include "gpu_info.h"
extern void update_density_fields_gpu(GPU_info *gpu_info, Phase *p);

extern void all_reduce_gpu(GPU_info *gpu_info,std::vector<int*> data , Phase *phase);
//extern void all_reduce(GPU_INFO *gpu_info,std::vector<int*> data , long size);
extern void reset_density_fields(GPU_info *gpu_info, Phase  *phase);
//void update_omega_fields_scmf0(GPU_INFO *gpu_info, Phase  *phase);
extern void reset_density_fields_back(GPU_info *gpu_info,Phase  *phase);

void update_omega_fields_scmf0(GPU_info *gpu_info, Phase  *phase);
#endif
