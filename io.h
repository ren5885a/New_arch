
#include "struct.h"
#include "gpu_info.h"
#include "mpi_info.h"


void io_out_density(MPI_info *mpi_info,GPU_info *gpu_info,Phase *phase,int step);

void io_out_aver_density(GPU_info *gpu_info,Phase *p,int step);

void io_out_coord(MPI_info *mpi_info,GPU_info *gpu_info,Phase *phase);
