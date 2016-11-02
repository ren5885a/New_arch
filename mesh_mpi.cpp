
#include "mesh_mpi.h"
#include <mpi.h>
void density_reset_mpi(Phase *phase,MPI_info *mpi_info,GPU_info *gpu_info){

	
	reset_density_fields(gpu_info,phase);
	int n_mono=0;
	for(int i=0;i<phase->n_cells*phase->n_mono_types;i++) n_mono+=phase->fields_unified[0][i];
	//printf("a node %d: %d\n",mpi_info->current_node,n_mono);

	long size=phase->n_cells*phase->n_mono_types;
	
	if (mpi_info->total_nodes > 1)

		MPI_Allreduce(MPI_IN_PLACE, phase->fields_unified[0], size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	 n_mono=0;
	for(int i=0;i<phase->n_cells*phase->n_mono_types;i++) n_mono+=phase->fields_unified[0][i];
	//printf("b node %d: %d\n",mpi_info->current_node,n_mono);
	
	 reset_density_fields_back(gpu_info,phase);
	  n_mono=0;
	//for(int i=0;i<phase->n_cells*2;i++) n_mono+=phase->fields_unified[1][i];
	//hprintf("c node %d: %d\n",mpi_info->current_node,n_mono);

}
