#include "io.h"


void io_out_density(MPI_info *mpi_info,GPU_info *gpu_info,Phase *phase,int step){


	if(mpi_info->current_node==0){
		
		FILE *dp;
		
		//int total_a=0;
		
		//for(int i=0;i<phase->n_cells;i++) {total_a=total_a+phase->fields_unified[0][i+phase->n_cells];}
		//printf("total particles %d\n",total_a);
		char name[20];
		if(step<1000000)
			sprintf(name,"density_%d.dat",step);
		else 
			sprintf(name,"density_%ge+06.dat",(double)step/1000000);
	
		dp=fopen(name,"w");
		if(dp==NULL) {
			printf("density.dat cannot open >_<, exit io_out_density.\n");
			return;
		}

		fprintf(dp,"Nx=%d, Ny=%d, Nz=%d\n",phase->nx,phase->ny,phase->nz);	
		fprintf(dp,"dx=%g, dy=%g, dz=%g\n",phase->dx,phase->dy,phase->dz);
	
		double N=0;
	
		for (unsigned int cell = 0; cell < phase->n_cells; cell++) {

			for (unsigned int T_types = 0; T_types < phase->n_mono_types; T_types++) {
		
				fprintf(dp,"%g ",phase->fields_unified[0][cell+T_types*phase->n_cells]*phase->field_scaling_type[T_types]);
			
			//*p->n_cells]*p->field_scaling_type[T_types]

			}
			//printf("\n ");
			for (unsigned int T_types = 0; T_types < phase->n_mono_types; T_types++) {
		
				fprintf(dp,"%g ",phase->omega_field_unified[0][cell+T_types*phase->n_cells]);
			

			}
			fprintf(dp,"\n");
			//printf("\n");

		}
	
		fclose(dp);
	}//end if node==0

}
void io_out_aver_density(GPU_info *gpu_info,Phase *p,int step){
	
	FILE *dp;
	char filename[100];

	if(step<1000000)
	sprintf(filename,"density_aver_%d.dat",step);
	else 
	sprintf(filename,"density_aver_%ge+06.dat",(double)step/1000000);
	dp=fopen(filename,"w+");

	fprintf(dp,"Nx=%d, Ny=%d, Nz=%d\n",p->nx,p->ny,p->nz);	
	fprintf(dp,"dx=%g, dy=%g, dz=%g\n",p->Lx/p->nx,p->Ly/p->ny,p->Lz/p->nz);
	
	for (unsigned int cell = 0; cell < p->n_cells; cell++) {

		for (unsigned int T_types = 0; T_types < p->n_mono_types; T_types++) {
		
			fprintf(dp,"%g ",p->average_field_unified[0][cell+T_types*p->n_cells]*p->field_scaling_type[T_types]);
			//if(cell==0) printf("%d %g %g\n",p->average_field_unified[0][cell+T_types*p->n_cells],p->field_scaling_type[T_types],p->average_field_unified[0][cell+T_types*p->n_cells]*p->field_scaling_type[T_types]);
			//*p->n_cells]*p->field_scaling_type[T_types]

		}
		//printf("\n ");
		for (unsigned int T_types = 0; T_types < p->n_mono_types; T_types++) {
		
			fprintf(dp,"%g ",p->omega_field_unified[0][cell+T_types*p->n_cells]);
			

		}
		fprintf(dp,"\n");
		//printf("\n");

	}
	
	fclose(dp);

}
void io_out_coord(MPI_info *mpi_info,GPU_info *gpu_info,Phase *phase){

	
	if(mpi_info->current_node==0){
	
		FILE *dp;

		dp=fopen("coord.dat","w");
	
		for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++)
	
			for(int bead=0;bead<phase->num_all_beads_per_gpu;bead++)
				fprintf(dp,"%g %g %g\n",phase->pos[gpu_index][bead*3],phase->pos[gpu_index][bead*3+1],phase->pos[gpu_index][bead*3+2]);
			

	}//end node 0

	else{
		for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++);
			//MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

	}

}// end routine



