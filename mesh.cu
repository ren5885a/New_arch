#include "mesh.h"
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include "device_function.cuh"

#include "init_config.h"
//!< sum up all the data in all GPU and save back to data int

extern void all_reduce_gpu(GPU_info *gpu_info,std::vector<int*> data , Phase *phase){


	int *temp;
	long size=phase->n_cells*phase->n_mono_types ;	

	checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[0]));	
	checkCudaErrors(cudaMallocManaged(&(temp),gpu_info->GPU_N*size*sizeof(int)));
	
	for (int i = 0; i < gpu_info->GPU_N; i++) {
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[i]));	
		checkCudaErrors( cudaDeviceSynchronize());
    		checkCudaErrors( cudaStreamSynchronize(gpu_info->stream[i]));
    
	}


	for(int gpu_index=1;gpu_index<gpu_info->GPU_N;gpu_index++){// gpu_info->GPU_N
		
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	

		checkCudaErrors(cudaMemcpyPeerAsync(temp+gpu_index*size, gpu_info->whichGPUs[0],data[gpu_index],gpu_info->whichGPUs[gpu_index],size*sizeof(int),gpu_info->stream[gpu_index]));

		checkCudaErrors( cudaDeviceSynchronize());

	}
	
	for (int i = 0; i < gpu_info->GPU_N; i++) {
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[i]));	
    		checkCudaErrors( cudaStreamSynchronize(gpu_info->stream[i]));
    
	}

	checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[0]));
	
	dim3 grid(gpu_info->gridNx,gpu_info->gridNy,gpu_info->gridNz);
	//printf("%d %d %d %d\n",grid.x,grid.y,grid.z,phase->MaxThreadDensity);
	//printf("%ld\n",size);
	reduce_field_int<<<grid,phase->MaxThreadDensity,0,gpu_info->stream[0]>>>(data[0],temp,phase->n_cells,gpu_info->GPU_N);
	
	checkCudaErrors( cudaDeviceSynchronize());

	

/*	

		
	

	for(int gpu_index=1;gpu_index<gpu_info->GPU_N;gpu_index++){// gpu_info->GPU_N
		
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	

		checkCudaErrors(cudaMemcpyPeerAsync(data[gpu_index],gpu_index,data[0], 0,size*sizeof(int),gpu_info->stream[gpu_index]));
		
		checkCudaErrors( cudaDeviceSynchronize());

	}
	
	for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){

		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	
		checkCudaErrors( cudaDeviceSynchronize());
		
		cudaStreamSynchronize(gpu_info->stream[gpu_index]);

	}
	checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[0]));	
	cudaFree(temp);
	*/
cudaFree(temp);
}

//!< sum up all the data in all GPU and save back to data double
extern void all_reduce_double(GPU_info *gpu_info,std::vector<double*> data,long size){


	double *temp;

	checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[0]));	
	checkCudaErrors(cudaMallocManaged(&(temp),gpu_info->GPU_N*size*sizeof(int)));
	
	for(int coord_index=0;coord_index<gpu_info->GPU_N*size;coord_index++){
		temp[coord_index]=0;

	}

	for(int gpu_index=1;gpu_index<gpu_info->GPU_N;gpu_index++){// gpu_info->GPU_N
		
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	

		checkCudaErrors(cudaMemcpyPeerAsync(temp+gpu_index*size, 0,data[gpu_index],gpu_index,size*sizeof(double),gpu_info->stream[gpu_index]));

	}


	

	for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){

		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	
		checkCudaErrors( cudaDeviceSynchronize());
		
		cudaStreamSynchronize(gpu_info->stream[gpu_index]);

	}
	
	int Nx,Ny,Nz;
	
	factor_decompose(gpu_info,size,&Nx,&Ny,&Nz);
	checkCudaErrors( cudaDeviceSynchronize());

	
	long N=0;
	for(int coord_index=0;coord_index<size;coord_index++){
		N+=temp[coord_index+size];

	}

	
	dim3 grid(Nx,Ny,Nz);
	checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[0]));
	reduce_field_double<<<grid,1>>>(data[0],temp,size,gpu_info->GPU_N);
	checkCudaErrors( cudaDeviceSynchronize());
	
	
	for(int gpu_index=1;gpu_index<gpu_info->GPU_N;gpu_index++){// gpu_info->GPU_N
		
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	

		checkCudaErrors(cudaMemcpyPeerAsync(data[gpu_index],gpu_index,data[0], 0,size*sizeof(double),gpu_info->stream[gpu_index]));
		
		checkCudaErrors( cudaDeviceSynchronize());

	}
	
	for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){

		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	
		checkCudaErrors( cudaDeviceSynchronize());
		
		cudaStreamSynchronize(gpu_info->stream[gpu_index]);

	}
	
	cudaFree(temp);


}


extern void reset_density_fields(GPU_info *gpu_info,Phase  *phase){

	//const unsigned int n_indices = phase->n_mono_types*phase->n_cells;
//coord_to_density(float *pos,int *density,Phase_info_gpu *phase_info_gpu,Poly_arch *poly_arch)

	for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){

		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	

		dim3 grid1(gpu_info->gridNx,gpu_info->gridNy,gpu_info->gridNz);
		//printf("density %d %d %d %d\n",grid1.x,grid1.y,grid1.z,phase->MaxThreadDensity);

		init_array<<<grid1,phase->MaxThreadDensity,0,gpu_info->stream[gpu_index]>>>(phase->fields_unified[gpu_index],phase->n_cells);

		checkCudaErrors( cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
		dim3 grid(gpu_info->polymerNx,gpu_info->polymerNy,gpu_info->polymerNz);
		//printf("Polymer %d %d %d %d\n",grid.x,grid.y,grid.z,phase->MaxThreadPolymer);
		coord_to_density<<<grid,phase->MaxThreadPolymer,0,gpu_info->stream[gpu_index]>>>(phase->pos[gpu_index],phase->fields_unified[gpu_index],phase->phase_info_gpu[gpu_index],phase->poly_arch[gpu_index]);

		checkCudaErrors( cudaDeviceSynchronize());
		
		cudaStreamSynchronize(gpu_info->stream[gpu_index]);

	}
	
	 all_reduce_gpu(gpu_info,phase->fields_unified , phase);
	
	
}
extern void reset_density_fields_back(GPU_info *gpu_info,Phase  *phase){

	int size=phase->n_cells*phase->n_mono_types;

	for(int gpu_index=1;gpu_index<gpu_info->GPU_N;gpu_index++){// gpu_info->GPU_N
		
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	

		checkCudaErrors(cudaMemcpyPeerAsync(phase->fields_unified[gpu_index],gpu_info->whichGPUs[gpu_index],phase->fields_unified[0], gpu_info->whichGPUs[0],size*sizeof(int),gpu_info->stream[gpu_index]));
		
		checkCudaErrors( cudaDeviceSynchronize());

	}
	
	for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){

		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	
		checkCudaErrors( cudaDeviceSynchronize());
		
		cudaStreamSynchronize(gpu_info->stream[gpu_index]);

	}
	checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[0]));	

}

void update_omega_fields_scmf0(GPU_info *gpu_info, Phase  *phase){

	//const double inverse_refbeads = 1.0 / phase->reference_Nbeads;

	
	
	int Nx=gpu_info->gridNx;
	int Ny=gpu_info->gridNy;
	int Nz=gpu_info->gridNz ;
	
	
	for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){// gpu_info->GPU_N
		
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	
		
		dim3 grid(Nx,Ny,Nz);
		
		
		
		
		
		omega_field_update<<<grid,phase->MaxThreadDensity,0,gpu_info->stream[gpu_index]>>>(phase->fields_unified[gpu_index], phase->omega_field_unified[gpu_index],phase->phase_info_gpu[gpu_index]);

		checkCudaErrors( cudaDeviceSynchronize());


	}//end loop for gpu_index

	
	
   
}// end routine update_omega_fields_scmf0
