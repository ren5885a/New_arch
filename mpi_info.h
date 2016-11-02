#ifndef MPI_INFO
#define MPI_INFO

#include<mpi.h>

#include "struct.h"
bool mpi_initialize(int argc,char *argv[],MPI_info *mpi_info);

extern void all_reduce_mpi(int *density,long length);
#endif//MPI_INFO
