//!< Initialize GPU device.
#include "device_function.cuh"
#include "gpu_info.h"
inline bool IsGPUCapableP2P(cudaDeviceProp *pProp)
{
#ifdef _WIN32
    return (bool)(pProp->tccDriver ? true : false);
#else
    return (bool)(pProp->major >= 2);
#endif
}


extern void init_cuda(MPI_info *mpi_info,GPU_info *gpu_info,int display){
	
	int gpu_count;
	int i,j;
	cudaDeviceProp prop[64];
	int *gpuid;
	int can_access_peer_0_1;
	
	gpu_count=0;
	gpuid=(int*)malloc(sizeof(int));

	if(gpu_info->GPU_N==0){

		//checkCudaErrors(cudaGetDeviceCount(&gpu_info->GPU_N));
	
		if(gpu_info->GPU_N==8)
			gpu_info->GPU_N=1;

		/*for(int i=0;i<gpu_info->GPU_N;i++)
			gpu_info->whichGPUs[i]=i;	//!Define on these GPU to calculate 
		*/

	}

	//gpu_info->whichGPUs=(int*)malloc(sizeof(int)*(gpu_info->GPU_N));

	gpu_info->stream=(cudaStream_t*)malloc(sizeof(cudaStream_t)*gpu_info->GPU_N);
	gpu_info->state.resize(gpu_info->GPU_N);
	
	printf("CPU %d:",mpi_info->current_node);

	
	for(i=0;i<(gpu_info->GPU_N);i++){
		
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[i]));
		checkCudaErrors(cudaStreamCreate ( &gpu_info->stream[i])) ;
		
		printf("%d ",i);//gpu_info->whichGPUs[i]
	}
	printf("is avilable \n");
	for (i=0; i < gpu_info->GPU_N; i++){
        	checkCudaErrors(cudaGetDeviceProperties(&gpu_info->prop[i], gpu_info->whichGPUs[i]));


		 

   		
   		

		
		// Only boards based on Fermi can support P2P
		
            	gpuid[gpu_count++] = gpu_info->whichGPUs[i];
		if(display==1){
			printf("> GPU%d = \"%15s\" %s capable of Peer-to-Peer (P2P)\n", i, gpu_info->prop[i].name, (IsGPUCapableP2P(&prop[i]) ? "IS " : "NOT"));
			printf("maxThreadsDim %d %d %d\n",gpu_info->prop[i].maxThreadsDim[0],gpu_info->prop[i].maxThreadsDim[1],gpu_info->prop[i].maxThreadsDim[2]);
            		printf("maxThreadsPerBlock %d\n",gpu_info->prop[i].maxThreadsPerBlock);
			printf("> GPU%d = \"%15s\" %s capable of Peer-to-Peer (P2P)\n", i, prop[i].name, (IsGPUCapableP2P(&prop[i]) ? "IS " : "NOT"));
			printf("> %s (GPU%d) supports UVA: %s\n", gpu_info->prop[i].name, i, (gpu_info->prop[i].unifiedAddressing ? "Yes" : "No"));
			
		}
		
		for(j=0;j<gpu_info->GPU_N;j++){
			if(i!=j){
				checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer_0_1, gpu_info->whichGPUs[i], gpu_info->whichGPUs[j]));
    				
				if(can_access_peer_0_1) {

					
					//checkCudaErrors(cudaDeviceEnablePeerAccess(gpu_info->whichGPUs[j], 0));
					
					
				}// if can_acesss				
				
				
			}//if i!=j
			

		}// for j
	
        }// for i
	
      free(gpuid);
   
}//end routine
