
#include "mpi_info.h"

bool mpi_initialize(int argc,char *argv[],MPI_info *mpi_info){

	MPI_Status status;
	MPI_Init(&argc, &argv);
	
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_info->current_node);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_info->total_nodes);	//MPI_Status status;

	return true;

}

extern void all_reduce_mpi(MPI_info *mpi_info,int *density,long length){

	if (mpi_info->total_nodes > 1)

		MPI_Allreduce(MPI_IN_PLACE, density,length, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	


}
