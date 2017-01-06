#include "test.h"
#include "gpu_info.h"
#include "init_config.h"

#include "mesh.h"
#include "mesh_mpi.h"
#include "smc.h"
#include "io.h"
extern void test_program(MPI_info *mpi_info,Phase *phase,int argc, char **argv){

	

	GPU_info gpu_info;
	

	init_all_config(&gpu_info,phase,mpi_info,argc, argv);


	
	checkCudaErrors(cudaGetLastError());
	density_reset_mpi(phase,mpi_info,&gpu_info);
	
	update_omega_fields_scmf0(&gpu_info, phase);

	//io_out_density(mpi_info,&gpu_info,phase,-20);
	

/*
	for(int i=0;i<32;i++){
		phase->pos[0][(1279968+i)*3]=i;	
		phase->pos[0][(1279968+i)*3+1]=0;
		phase->pos[0][(1279968+i)*3+2]=0;
	}		
	
*/

	for(int step=phase->start_time+1;step<=(phase->N_steps+phase->start_time);step++){

		smc_move( phase,&gpu_info,mpi_info,1);
		density_reset_mpi(phase,mpi_info,&gpu_info);
		
		if(step%10) 
		update_omega_fields_scmf0(&gpu_info, phase);

		if(step%phase->Tcheck==0){

		double end_dis=0;

		for(int gpu_index=0;gpu_index<gpu_info.GPU_N;gpu_index++){
			for(int polymer_index=0;polymer_index<phase->n_polymers_per_gpu;polymer_index++){
				double x=phase->pos[gpu_index][polymer_index*32*3];
				double y=phase->pos[gpu_index][polymer_index*32*3+1];
				double z=phase->pos[gpu_index][polymer_index*32*3+2];
			

				double x1=phase->pos[gpu_index][(polymer_index*32+23)*3];
				double y1=phase->pos[gpu_index][(polymer_index*32+23)*3+1];
				double z1=phase->pos[gpu_index][(polymer_index*32+23)*3+2];

				end_dis+=(x-x1)*(x-x1)+(y-y1)*(y-y1)+(z-z1)*(z-z1);
			
				//if(polymer_index%1000==0&&gpu_index==1) printf("%g %g %g\n",x,y,z);	

			}

		}// end for gpu_index
		end_dis=end_dis/(gpu_info.GPU_N*phase->n_polymers_per_gpu);
		printf("step %d end to end dis %g\n",step,end_dis);
		io_out_density(mpi_info,&gpu_info,phase,step);
		}


	}
	
		//for(int i=0;i<32;i++) printf("%g \n",phase->pos[0][32*3+i]);

	
	io_out_coord(mpi_info,&gpu_info,phase);
	float time;
	//printf("begining %g %g %g\n",phase->pos[0][0],phase->pos[0][1],phase->pos[0][2]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	smc_move( phase,&gpu_info,mpi_info,1);
	density_reset_mpi(phase,mpi_info,&gpu_info);
	update_omega_fields_scmf0(&gpu_info, phase);
	//printf("programm end here %d\n",mpi_info->current_node);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("time =%g\n",time);
	
	//printf("end %g %g %g\n",phase->pos[0][0],phase->pos[0][1],phase->pos[0][2]);
	//update_omega_fields_scmf0(&gpu_info, phase);

//--------------------------test code for Phase_info_gpu --------------------------------------------------------
	
	//test_Phase_info_gpu(0,0,phase,&gpu_info,mpi_info);

// -----------------------test code for end to end distance------------------------------------------------------
/*	

	double end_dis=0;

	for(int gpu_index=0;gpu_index<gpu_info.GPU_N;gpu_index++){
		for(int polymer_index=0;polymer_index<phase->n_polymers_per_gpu;polymer_index++){
			double x=phase->pos[gpu_index][polymer_index*32*3];
			double y=phase->pos[gpu_index][polymer_index*32*3+1];
			double z=phase->pos[gpu_index][polymer_index*32*3+2];
			

			double x1=phase->pos[gpu_index][(polymer_index*32+31)*3];
			double y1=phase->pos[gpu_index][(polymer_index*32+31)*3+1];
			double z1=phase->pos[gpu_index][(polymer_index*32+31)*3+2];

			end_dis+=(x-x1)*(x-x1)+(y-y1)*(y-y1)+(z-z1)*(z-z1);
			
			//if(polymer_index%1000==0&&gpu_index==1) printf("%g %g %g\n",x,y,z);	

		}

	}// end for gpu_index

	
	
	end_dis=end_dis/(gpu_info.GPU_N*phase->n_polymers_per_gpu);
	printf("end to end dis %g\n",end_dis);

*/
	
}

