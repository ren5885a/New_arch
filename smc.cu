#include "smc.h"

extern void smc_move( Phase *phase,GPU_info *gpu_info,MPI_info *mpi_info,int step){


	

	for(int i=0;i<step;i++){
		
		for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){

			checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	

		
			dim3 grid(gpu_info->polymerNx,gpu_info->polymerNy,gpu_info->polymerNz);

			//printf("cite gpu %d %d %d %d\n",grid.x,grid.y,grid.z,phase->MaxThreadPolymer);
			//grid,phase->MaxThreadPolymer,0,gpu_info->stream[gpu_index]
			

			mc_polymer_move_arr<<<grid,phase->MaxThreadPolymer,0,gpu_info->stream[gpu_index]>>>(phase->phase_info_gpu[gpu_index],phase->poly_arch[gpu_index],phase->pos[gpu_index],gpu_info->state[gpu_index],phase->omega_field_unified[gpu_index]);

			checkCudaErrors( cudaDeviceSynchronize());

			//checkCudaErrors(cudaGetLastError());
		
			//cudaStreamSynchronize(gpu_info->stream[gpu_index]);

			/*
			if(i%100==1&&gpu_index==0){	

				double end_dis=0;

		
				for(int polymer_index=0;polymer_index<phase->n_polymers_per_gpu;polymer_index++){
		
					double x=phase->pos[gpu_index][polymer_index*32*3];
					double y=phase->pos[gpu_index][polymer_index*32*3+1];
					double z=phase->pos[gpu_index][polymer_index*32*3+2];
			

					double x1=phase->pos[gpu_index][(polymer_index*32+31)*3];
					double y1=phase->pos[gpu_index][(polymer_index*32+31)*3+1];
					double z1=phase->pos[gpu_index][(polymer_index*32+31)*3+2];

					end_dis+=(x-x1)*(x-x1)+(y-y1)*(y-y1)+(z-z1)*(z-z1);
			
				

				}//end for polymer_index

				end_dis=end_dis/(phase->n_polymers_per_gpu);
				printf(" %d end to end dis %g on gpu %d\n",i,end_dis,gpu_index);
			}//end if step
	
			*/
			
			

			
	
	
		}//end gpu_index


		

	
	}//loop end step

	
}// smc_move end 

extern void test_stream(GPU_info *gpu_info){

	dim3 grid(2,2,20);

	double *pa;
	cudaStream_t stream1,stream2,stream3,stream4;
	checkCudaErrors(cudaSetDevice(0));	
	checkCudaErrors(cudaMalloc(&(pa),10000 ));
	

	checkCudaErrors( cudaDeviceSynchronize());
	cudaStreamCreate ( &stream1) ;
	cudaStreamCreate ( &stream2) ;
	cudaStreamCreate ( &stream3) ;
	cudaStreamCreate ( &stream4) ;
	cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream2,cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream3,cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream4,cudaStreamNonBlocking);
		checkCudaErrors(cudaSetDevice(0));
		test_stream<<<grid,1024,0,stream1>>>(pa);
		
		test_stream<<<grid,1024,0,stream2>>>(pa);
		
		test_stream<<<grid,1024,0,stream3>>>(pa);
		
		test_stream<<<grid,1024,0,stream4>>>(pa);
cudaStreamSynchronize ( stream1 );
cudaStreamSynchronize ( stream2 );
cudaStreamSynchronize ( stream3 );
cudaStreamSynchronize ( stream4 );
checkCudaErrors(cudaSetDevice(0));
checkCudaErrors( cudaDeviceSynchronize());

	
	
	
}


extern void test_Phase_info_gpu(int cpu_index,int gpu_index,Phase *phase,GPU_info *gpu_info,MPI_info *mpi_info){

	if(mpi_info->current_node==cpu_index){

		printf("----------Program phase data on node %d GPU %d is tested! \n",cpu_index,gpu_index);
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	
		printf("chi N matrix:\n");
		for(int i=0;i<phase->n_mono_types;i++){
			for(int j=0;j<phase->n_mono_types;j++)
				printf("%g ",phase->phase_info_gpu[gpu_index]->xn[i+j*phase->n_mono_types]);

			printf("\n");
		}
		
		printf("nx %d ny %d nz %d\n",phase->phase_info_gpu[gpu_index]->nx,phase->phase_info_gpu[gpu_index]->ny,phase->phase_info_gpu[gpu_index]->nz);
		printf("total cells %ld \n",phase->phase_info_gpu[gpu_index]->n_cells);

		printf("Lx %g Ly %g Lz %g\n",phase->phase_info_gpu[gpu_index]->Lx,phase->phase_info_gpu[gpu_index]->Ly,phase->phase_info_gpu[gpu_index]->Lz);
		printf("iLx %g iLy %g iLz %g\n",phase->phase_info_gpu[gpu_index]->iLx,phase->phase_info_gpu[gpu_index]->iLy,phase->phase_info_gpu[gpu_index]->iLz);
		printf("dx %g dy %g dz %g\n",phase->phase_info_gpu[gpu_index]->dx,phase->phase_info_gpu[gpu_index]->dy,phase->phase_info_gpu[gpu_index]->dz);

		printf("n_polymers: %d \n",phase->phase_info_gpu[gpu_index]->n_polymers);
		printf("n_polymers per node : %d \n",phase->phase_info_gpu[gpu_index]->n_polymers_per_node);
		printf("n_polymers per gpu : %d \n",phase->phase_info_gpu[gpu_index]->n_polymers_per_gpu);
		
		printf("polymer_type_number: %d \n",phase->phase_info_gpu[gpu_index]->polymer_type_number);
		
		printf("n_polymer_type: ");
		for(int i=0;i<phase->phase_info_gpu[gpu_index]->polymer_type_number;i++)printf("%d ",phase->phase_info_gpu[gpu_index]->n_polymer_type[i]);
		printf("\n");

		printf("n_polymer_type_node: ");
		for(int i=0;i<phase->phase_info_gpu[gpu_index]->polymer_type_number;i++)printf("%d ",phase->phase_info_gpu[gpu_index]->n_polymers_type_per_node[i]);
		printf("\n");

		printf("n_polymer_type_gpu: ");
		for(int i=0;i<phase->phase_info_gpu[gpu_index]->polymer_type_number;i++)printf("%d ",phase->phase_info_gpu[gpu_index]->n_polymers_type_per_gpu[i]);
		printf("\n");
		
		printf("n_mono_types %d \n",phase->phase_info_gpu[gpu_index]->n_mono_types);
		printf("num_all_beads: %d \n",phase->phase_info_gpu[gpu_index]->num_all_beads);
		printf("num_all_beads_per_node: %d \n",phase->phase_info_gpu[gpu_index]->num_all_beads_per_node);
		printf("num_all_beads_per_gpu: %d \n",phase->phase_info_gpu[gpu_index]->num_all_beads_per_gpu);

		printf("n_bead_type_gpu: ");
		for(int i=0;i<phase->phase_info_gpu[gpu_index]->n_mono_types;i++)printf("%d ",phase->phase_info_gpu[gpu_index]->num_bead_type[i]);
		printf("\n");

		printf("num_all_beads_per_node: ");
		for(int i=0;i<phase->phase_info_gpu[gpu_index]->n_mono_types;i++)printf("%d ",phase->phase_info_gpu[gpu_index]->num_bead_type_per_node[i]);
		printf("\n");

		printf("num_all_beads_per_gpu: ");
		for(int i=0;i<phase->phase_info_gpu[gpu_index]->n_mono_types;i++)printf("%d ",phase->phase_info_gpu[gpu_index]->num_bead_type_per_gpu[i]);
		printf("\n");

		printf("polymer_basis_gpu: ");
		for(int i=0;i<=phase->phase_info_gpu[gpu_index]->polymer_type_number;i++)printf("%d ",phase->phase_info_gpu[gpu_index]->polymer_basis_gpu[i]);
		printf("\n");

		printf("bead_basis_gpu: ");
		for(int i=0;i<=phase->phase_info_gpu[gpu_index]->n_mono_types;i++)printf("%d ",phase->phase_info_gpu[gpu_index]->monomer_poly_basis_gpu[i]);
		printf("\n");

		printf("field_scaling_type: ");
		for(int i=0;i<=phase->phase_info_gpu[gpu_index]->n_mono_types;i++)printf("%g ",phase->phase_info_gpu[gpu_index]->field_scaling_type[i]);
		printf("\n");
		
		printf("inverse_refbeads: %g\n",phase->phase_info_gpu[gpu_index]->inverse_refbeads);
		printf("refbeads: %d\n",phase->phase_info_gpu[gpu_index]->reference_Nbeads);
		printf("harmonic_normb: %g\n",phase->phase_info_gpu[gpu_index]->harmonic_normb);

		printf("current_node: %d\n",phase->phase_info_gpu[gpu_index]->current_node);
		printf("total_nodes: %d\n",phase->phase_info_gpu[gpu_index]->total_nodes);
		printf("MaxThreadDensity: %d\n",phase->phase_info_gpu[gpu_index]->MaxThreadDensity);
		printf("MaxThreadPolymer: %d\n",phase->phase_info_gpu[gpu_index]->MaxThreadPolymer);


		printf("----------Program structure data on node %d GPU %d is tested! \n",cpu_index,gpu_index);

		for(int poly_type=0;poly_type<phase->phase_info_gpu[gpu_index]->polymer_type_number;poly_type++){

			printf("poly index %d \n",poly_type);
			printf("poly_length %d \n",phase->poly_arch[gpu_index][poly_type].poly_length);
			
			for(int i=0;i<phase->phase_info_gpu[gpu_index]->n_mono_types;i++)
				printf("%d ",phase->poly_arch[gpu_index][poly_type].mono_type_length[i]);
			printf("\n");

			
			printf("length_bond %g\n",phase->poly_arch[gpu_index][poly_type].length_bond);

			printf("Mono type\n");
			for(int i=0;i<phase->poly_arch[gpu_index][poly_type].poly_length;i++)
				printf("%d ",phase->poly_arch[gpu_index][poly_type].Monotype[i]);
			printf("\n");
			printf("\n");
			printf("Structure of polymer!\n");
			for(int i=0;i<phase->poly_arch[gpu_index][poly_type].poly_length;i++){
				for(int j=0;j<phase->poly_arch[gpu_index][poly_type].poly_length;j++)
	
					printf("%d ",phase->poly_arch[gpu_index][poly_type].connection[i+j*phase->poly_arch[gpu_index][poly_type].poly_length]);

				printf("\n");

			}
			printf("\n");
			for(int i=0;i<phase->poly_arch[gpu_index][poly_type].poly_length;i++) printf("%d ",phase->poly_arch[gpu_index][poly_type].neigh_num[i]);
			printf("\n");
			
			printf("reference_Nbeads %d \n",phase->poly_arch[gpu_index][poly_type].reference_Nbeads);

			for(int i=0;i<phase->poly_arch[gpu_index][poly_type].poly_length;i++){
				for(int j=0;j<phase->poly_arch[gpu_index][poly_type].neigh_num[i];j++)
					printf("%d ",phase->poly_arch[gpu_index][poly_type].conection_list[i][j]);

				printf("\n");

			}
			
			printf("--------------------\n");

		}// end for poly_type

	}//end if cpu chosen

	


}
