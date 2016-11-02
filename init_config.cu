#include "init_config.h"

// decompose a integer like number of grid points and number of polymer into three integer  Nx Ny Nz to suit cuda//

int prime[168]={2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997};

// find the maximal factor in a integer which is smaller than 1024(Maximal thread number in cuda)
extern int factor_decompose_1024(GPU_info *gpu_info,long N){

	long temp;
	
	temp=N;

	int decom[10000],index=0;
	
	for(int i=0;i<168;i++){
		
		while(temp%prime[i]==0){
			temp=temp/prime[i];
			decom[index++]=prime[i];

		};
		
		

	}
	
	int *elements;

	elements=(int*)malloc(sizeof(int)*index);
	
	for(int i=0;i<index;i++) elements[i]=0;
	
	int temp_1024=1;

	for(int j=1;j<=10;j++){
		elements[j-1]=1;
  		const size_t N_t = index;
  	
  		std::vector<int> selectors(elements, elements + N_t);
	
  		
  		do{
			int combo=1;
			for (size_t i = 0; i < selectors.size(); ++i){
      				if (selectors[i]){
        				//std::cout << decom[i] << ", ";
					combo*=decom[i];
     				 }
   			}
		
			if(combo>temp_1024&&combo<=1024) temp_1024=combo;
			if(combo==1024) break;
			
  		} while (prev_permutation(selectors.begin(), selectors.end()));


	}
	
	free(elements);
	return temp_1024;
	
	
}

extern void factor_decompose(GPU_info *gpu_info,long N, int *Nx_a,int *Ny_a,int *Nz_a){

	
	int Nx,Ny,Nz;
	long temp;
	
	temp=N;

	int decom[10000],index=0;
	for(int i=0;i<168;i++){
		
		while(temp%prime[i]==0){
			temp=temp/prime[i];
			decom[index++]=prime[i];

		};
		
		

	}
	//printf("%ld prime is ",N);
	//for(int i=0;i<index;i++) printf(" %d ",decom[i]);
	//printf("\n");

	if(temp!=1) {
			
		printf("please give a \"good\" polymer number!\n");
		exit(0);
	}

	if(index==1) {
		
		Nx=N;
		Ny=1;
		Nz=1;
		
		

	}
	else if(index==2){
		
		Nz=1;//decom[index-1]
		Ny=decom[0];
		Nx=decom[1];
		//printf("%d %d\n",Nx,Ny);
		
	}
	else if(index>2){
		
		Nx=1;
		Ny=1;
		Nz=1;
		if(index%2==0){
			
			Nz=decom[index-1]*decom[0];

			if((index-2)%4==0){

				for(int i=0;i<(index-2)/4;i++){
					Nx*=decom[i+1]*decom[index-1-i-1];
					Ny*=decom[(index-2)/4+1+i]*decom[index-1-(index-2)/4-1-i];
					
				}
				//printf("%d %d %d\n",Nx,Ny,Nz);

			}
			else if((index-2)==2){
				
				Ny=decom[1];
				Nx=decom[2];
				//printf("%d %d %d\n",Nx,Ny,Nz);

			}
			else {
				Nz*=decom[1]*decom[2];
				for(int i=0;i<(index-4)/4;i++){

					Nx*=decom[i+3]*decom[index-1-i-1];
					Ny*=decom[(index-2)/4+3+i]*decom[index-1-(index-2)/4-1-i];
				}
				//printf("%d %d %d\n",Nx,Ny,Nz);
		
			}


		}
		else{
			Nz=decom[index-1];
			if((index-1)%4==0){

				for(int i=0;i<(index-1)/4;i++){

					Nx*=decom[i]*decom[index-1-i-1];
					Ny*=decom[(index-1)/4+i]*decom[index-1-(index-1)/4-i-1];
					
				}
				//printf("%d: %d %d %d\n",index,Nx,Ny,Nz);

			}
			else if((index-1)==2){
				
				Ny=decom[0];
				Nx=decom[1];
				//printf("%d %d %d\n",Nx,Ny,Nz);

			}
			else {
				Nz*=decom[0]*decom[1];
				for(int i=0;i<(index-3)/4;i++){

					Nx*=decom[i*2+2]*decom[index-1-i*2-1];
					Ny*=decom[i*2+3]*decom[index-3-i*2];
				}
				//printf("%d %d %d\n",Nx,Ny,Nz);
		
			}


		}
		
	}
	if(N==1) {
		Nx=1;
		Ny=1;
		Nz=1;

	}
	
	if(Nx*Ny*Nz==N) {
		
		*Nx_a=Nx;
		*Ny_a=Ny;
		*Nz_a=Nz;

	}
	else {

		printf("Error Nx %d *Ny %d  *Nz %d!= N %ld\n",Nx,Ny,Nz,N);
		exit(0);
	}
}


//<! Read in CPU and GPU polymer structure infomation in phase->poly_arch
extern void Read_polymer_config(MPI_info *mpi_info,GPU_info *gpu_info,Phase *phase){

	FILE *dp;
	
	dp=fopen("polymer.dat","r");

	fscanf(dp,"Number of polymer type:%d\n",&phase->polymer_type_number);
	
	fscanf(dp,"Number of total polymer:%d\n",&phase->n_polymers);

	phase->n_polymers_per_node=phase->n_polymers/mpi_info->total_nodes;
	
	phase->n_polymers_per_gpu=phase->n_polymers_per_node/gpu_info->GPU_N;

	//printf("total: %d  Node: %d GPU: %d\n",phase->n_polymers,phase->n_polymers_per_node,phase->n_polymers_per_gpu);
	
	phase->poly_arch.resize(gpu_info->GPU_N);

	
	phase->n_polymer_type=(unsigned int *)malloc(sizeof(unsigned int)*phase->polymer_type_number);
	phase->n_polymers_type_per_node=(unsigned int *)malloc(sizeof(unsigned int)*phase->polymer_type_number);
	phase->n_polymers_type_per_gpu=(unsigned int *)malloc(sizeof(unsigned int)*phase->polymer_type_number);

	for(int i=0;i<phase->polymer_type_number;i++){
	
		fscanf(dp,"%d ",&phase->n_polymer_type[i]);
		phase->n_polymers_type_per_node[i]=phase->n_polymer_type[i]/mpi_info->total_nodes;
		phase->n_polymers_type_per_gpu[i]=phase->n_polymers_type_per_node[i]/gpu_info->GPU_N;
	}

	

	fscanf(dp,"\n--------\n");

	
	if(mpi_info->current_node==0)
	for(int i=0;i<phase->polymer_type_number;i++){
		//printf("Type %d :node: %d GPU: %d\n",i,phase->n_polymers_type_per_node[i],phase->n_polymers_type_per_gpu[i]);
	}

	
	
	for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){ //

		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	
		
		checkCudaErrors(cudaMallocManaged(&phase->poly_arch[gpu_index],phase->polymer_type_number));
		
		if(gpu_index==0){

			for(int polymer_index=0;polymer_index<phase->polymer_type_number;polymer_index++){
		
				
				int temp;
				fscanf(dp,"Polymer type: %d\n",&temp);//phase->poly_arch[gpu_index][polymer_index].polymer_type_index

				fscanf(dp,"Polymer length: %d\n",&phase->poly_arch[gpu_index][polymer_index].poly_length);
				fscanf(dp,"Polymer length unit: %d\n",&phase->reference_Nbeads);
				
				
				checkCudaErrors(cudaMallocManaged(&phase->poly_arch[gpu_index][polymer_index].Monotype,phase->poly_arch[gpu_index][polymer_index].poly_length));
		
				fscanf(dp,"Monomer type: ");
				
				
				
				for(int i=0;i<phase->poly_arch[gpu_index][polymer_index].poly_length;i++){
			
					fscanf(dp,"%d ",&phase->poly_arch[gpu_index][polymer_index].Monotype[i]);
					
					//phase->poly_arch[gpu_index][polymer_index].mono_type_length[phase->poly_arch[gpu_index][polymer_index].Monotype[i]]++;
				}
				
				int poly_length=phase->poly_arch[gpu_index][polymer_index].poly_length;

				fscanf(dp,"\n");		
				
				checkCudaErrors(cudaMallocManaged(&phase->poly_arch[gpu_index][polymer_index].connection,poly_length*poly_length));

				
		
				for(int i=0;i<poly_length;i++){

					for(int j=0;j<poly_length;j++){
	
						fscanf(dp,"%d ",&phase->poly_arch[gpu_index][polymer_index].connection[i+j*poly_length]);
		
					}//! end for j

					fscanf(dp,"\n");
			

				}//! end for i
				
				checkCudaErrors(cudaMallocManaged(&phase->poly_arch[gpu_index][polymer_index].neigh_num,poly_length));

				checkCudaErrors(cudaMallocManaged(&phase->poly_arch[gpu_index][polymer_index].conection_list,poly_length));
	
				
		
				int index;
				
				for(int i=0;i<poly_length;i++){
	
					phase->poly_arch[gpu_index][polymer_index].neigh_num[i]=0;
			
					for(int j=0;j<poly_length;j++){
				
						if(phase->poly_arch[gpu_index][polymer_index].connection[i+j*poly_length]==1){
				
							phase->poly_arch[gpu_index][polymer_index].neigh_num[i]++;
			
						}//!< end if connnected
		
						checkCudaErrors(cudaMallocManaged(&phase->poly_arch[gpu_index][polymer_index].conection_list[i],phase->poly_arch[gpu_index][polymer_index].neigh_num[i]));
				
					}//!< for loop j
		
					index=0;
		
					for(int j=0;j<poly_length;j++){
				
						if(phase->poly_arch[gpu_index][polymer_index].connection[i+j*poly_length]==1)
	
							phase->poly_arch[gpu_index][polymer_index].conection_list[i][index++]=j;				

					}
			


				}//!< for loop i
				
			}//! end for polymer index

			phase->num_all_beads=0;
			phase->num_all_beads_per_node=0;
			phase->num_all_beads_per_gpu=0;

			phase->num_bead_polymer_type=(unsigned int *)malloc(sizeof(unsigned int)*phase->polymer_type_number);
			phase->num_bead_polymer_type_per_node=(unsigned int *)malloc(sizeof(unsigned int)*phase->polymer_type_number);
			phase->num_bead_polymer_type_per_gpu=(unsigned int *)malloc(sizeof(unsigned int)*phase->polymer_type_number);

			phase->num_bead_type=(unsigned int *)malloc(sizeof(unsigned int)*phase->n_mono_types);
			phase->num_bead_type_per_node=(unsigned int *)malloc(sizeof(unsigned int)*phase->n_mono_types);
			phase->num_bead_type_per_gpu=(unsigned int *)malloc(sizeof(unsigned int)*phase->n_mono_types);


			for(int polymer_index=0;polymer_index<phase->polymer_type_number;polymer_index++){

				int length=phase->poly_arch[gpu_index][polymer_index].poly_length;

				phase->num_bead_polymer_type[polymer_index]=length*phase->n_polymer_type[polymer_index];
				phase->num_bead_polymer_type_per_node[polymer_index]=length*phase->n_polymers_type_per_node[polymer_index];
				phase->num_bead_polymer_type_per_gpu[polymer_index]=length*phase->n_polymers_type_per_gpu[polymer_index];
			
				phase->num_all_beads+=length*phase->n_polymer_type[polymer_index];
				phase->num_all_beads_per_node+=length*phase->n_polymers_type_per_node[polymer_index];
				phase->num_all_beads_per_gpu+=length*phase->n_polymers_type_per_gpu[polymer_index];
			}
			
	
		}// ! end if (gpu_index==0)
		else{
			for(int polymer_index=0;polymer_index<phase->polymer_type_number;polymer_index++){

				phase->poly_arch[gpu_index][polymer_index].polymer_type_index=phase->poly_arch[0][polymer_index].polymer_type_index;
		
				phase->poly_arch[gpu_index][polymer_index].poly_length=phase->poly_arch[0][polymer_index].poly_length;

				checkCudaErrors(cudaMallocManaged(&phase->poly_arch[gpu_index][polymer_index].Monotype,phase->poly_arch[gpu_index][polymer_index].poly_length));

				for(int i=0;i<phase->poly_arch[gpu_index][polymer_index].poly_length;i++){
			
						phase->poly_arch[gpu_index][polymer_index].Monotype[i]=phase->poly_arch[0][polymer_index].Monotype[i];

				}
			
				int poly_length=phase->poly_arch[gpu_index][polymer_index].poly_length;

				checkCudaErrors(cudaMallocManaged(&phase->poly_arch[gpu_index][polymer_index].connection,poly_length*poly_length));

			
				for(int i=0;i<poly_length;i++){

					for(int j=0;j<poly_length;j++){
	
						phase->poly_arch[gpu_index][polymer_index].connection[i+j*poly_length]=phase->poly_arch[0][polymer_index].connection[i+j*poly_length];
		
					}//! end for j

					
			

				}//! end for i


				checkCudaErrors(cudaMallocManaged(&phase->poly_arch[gpu_index][polymer_index].neigh_num,poly_length));

				checkCudaErrors(cudaMallocManaged(&phase->poly_arch[gpu_index][polymer_index].conection_list,poly_length));


				int index;
		
				for(int i=0;i<poly_length;i++){
		
					phase->poly_arch[gpu_index][polymer_index].neigh_num[i]=0;
				
					for(int j=0;j<poly_length;j++){
				
						if(phase->poly_arch[gpu_index][polymer_index].connection[i+j*poly_length]==1){
				
							phase->poly_arch[gpu_index][polymer_index].neigh_num[i]++;
			
						}//!< end if connnected
		
						checkCudaErrors(cudaMallocManaged(&phase->poly_arch[gpu_index][polymer_index].conection_list[i],phase->poly_arch[gpu_index][polymer_index].neigh_num[i]));
				
					}//!< for loop j
		
					index=0;
		
					for(int j=0;j<poly_length;j++){
				
						if(phase->poly_arch[gpu_index][polymer_index].connection[i+j*poly_length]==1)
	
							phase->poly_arch[gpu_index][polymer_index].conection_list[i][index++]=j;				

					}
			


				}//!< for loop i

			}// end for polymer
		} // end if gpu_index ==0




	}	

}



extern void initialize_values(GPU_info *gpu_info,Phase *phase){

	phase->start_clock = time(NULL);
	phase->n_accepts = 0;
	phase->n_moves =0;

	phase->GPU_N=gpu_info->GPU_N;
	phase->gridNx=gpu_info->gridNx;
	phase->gridNy=gpu_info->gridNy;
	phase->gridNz=gpu_info->gridNz;

	phase->polymerNx=gpu_info->polymerNx;
	phase->polymerNy=gpu_info->polymerNy;
	phase->polymerNz=gpu_info->polymerNz;
	
	
	phase->ana_info.delta_mc_Re=10;
	phase->ana_info.filename = (char*)malloc( 200*sizeof(char) );
	sprintf(phase->ana_info.filename,"Re.dat");
	// Reference Harmonic Spring Cste

	for(int i=0;i<gpu_info->GPU_N;i++)
		for(int j=0;j<phase->polymer_type_number;j++)
	phase->poly_arch[i][j].reference_Nbeads=phase->reference_Nbeads;
	
	phase->harmonic_spring_Cste =1.0 / sqrt(3.0 * (phase->reference_Nbeads - 1.0));
	
	//Reference energy scale for harmonic springs.
	
	phase->harmonic_normb =1.0 / (2.0 * phase->harmonic_spring_Cste * phase->harmonic_spring_Cste);

	
	FILE *dp;
	dp=fopen("configuration.dat","r");
	if(dp==NULL) {printf("Empty pointer allocate at line number %d in file %s\n", __LINE__, __FILE__);exit(0);}

	fscanf(dp,"Monomer type number: %d\n",&phase->n_mono_types);

	phase->xn=(double **)malloc(sizeof(double *)*phase->n_mono_types);
	for(int i=0;i<phase->n_mono_types;i++) phase->xn[i]=(double *)malloc(sizeof(double )*phase->n_mono_types);
	
	for(int j=0;j<phase->n_mono_types;j++){
		for(int i=0;i<phase->n_mono_types;i++) {
			fscanf(dp,"%lg ",&phase->xn[i][j]);
		}
		fscanf(dp,"\n");
	}
	
	fscanf(dp,"D0 =     %lg; LY =    %lg; LZ =    %lg;\n",&phase->Lx,&phase->Ly,&phase->Lz);
	fscanf(dp,"N = %d %d %d \n",&phase->nx,&phase->ny,&phase->nz);

	phase->n_cells = phase->nx * phase->ny * phase->nz;

	fscanf(dp,"time =  %d;\n",&phase->time);
	//phase->start_time = phase->time;
	fscanf(dp,"xiN = %lg;\n",&phase->xiN);
	fscanf(dp,"dt*N/xi = %lg;\n",&phase->dtNxi);
	fscanf(dp,"kT = %lg;\n",&phase->kT);
	//fscanf(dp,"TCHECK = %d;\n",&phase->Tcheck);
	//fscanf(dp,"TWRITE = %d\n;",&phase->Twrite);

	//MPI_Allreduce(&(p->n_polymers), &n_polymers_global_sum, 1,MPI_UNSIGNED, MPI_SUM, info_MPI->SOMA_MPI_Comm);
	//assert(p->n_polymers_global == n_polymers_global_sum);
	
	phase->area51.resize(gpu_info->GPU_N);
	phase->phase_info_gpu.resize(gpu_info->GPU_N);
	phase->external_field_unified.resize(gpu_info->GPU_N);
	phase->umbrella_field_unified.resize(gpu_info->GPU_N);
	phase->average_field_unified.resize(gpu_info->GPU_N);
	phase->temp_average_field_unified.resize(gpu_info->GPU_N);
	

	phase->MaxThreadDensity=factor_decompose_1024(gpu_info,phase->nx*phase->ny*phase->nz);
	phase->MaxThreadPolymer=factor_decompose_1024(gpu_info,phase->n_polymers_per_gpu);
	
	//printf("phase->MaxThreadDensity=%d phase->MaxThreadPolymer=%d\n",phase->MaxThreadDensity,phase->MaxThreadPolymer);
	
	for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){// gpu_info->GPU_N
		
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	
		
		phase->area51[gpu_index]=NULL;

			
		phase->external_field_unified[gpu_index]=NULL;

		
		phase->umbrella_field_unified[gpu_index]=NULL;

		
		phase->average_field_unified[gpu_index]=NULL;

		
		phase->temp_average_field_unified[gpu_index]=NULL;

		checkCudaErrors(cudaMallocManaged((void**)&(phase->phase_info_gpu[gpu_index]),sizeof(Phase_info_gpu)));

		phase->phase_info_gpu[gpu_index]->nx=phase->nx;
		phase->phase_info_gpu[gpu_index]->ny=phase->ny;
		phase->phase_info_gpu[gpu_index]->nz=phase->nz;
		phase->phase_info_gpu[gpu_index]->n_cells=phase->nx*phase->ny*phase->nz;
		//printf("%d: %d %d %d\n",phase->phase_info_gpu[gpu_index]->n_cells,phase->phase_info_gpu[gpu_index]->nx,phase->phase_info_gpu[gpu_index]->ny,phase->phase_info_gpu[gpu_index]->nz);
		phase->phase_info_gpu[gpu_index]->Lx=phase->Lx;
		phase->phase_info_gpu[gpu_index]->Ly=phase->Ly;
		phase->phase_info_gpu[gpu_index]->Lz=phase->Lz;

		phase->phase_info_gpu[gpu_index]->iLx=1/phase->Lx;
		phase->phase_info_gpu[gpu_index]->iLy=1/phase->Ly;
		phase->phase_info_gpu[gpu_index]->iLz=1/phase->Lz;

		phase->phase_info_gpu[gpu_index]->dx=phase->Lx/phase->nx;
		phase->phase_info_gpu[gpu_index]->dy=phase->Ly/phase->ny;
		phase->phase_info_gpu[gpu_index]->dz=phase->Lz/phase->nz;

		phase->phase_info_gpu[gpu_index]->polymer_type_number=phase->polymer_type_number;
		phase->phase_info_gpu[gpu_index]->n_polymers=phase->n_polymers;
		phase->phase_info_gpu[gpu_index]->n_polymers_per_node=phase->n_polymers_per_node;
		phase->phase_info_gpu[gpu_index]->n_polymers_per_gpu=phase->n_polymers_per_gpu;
		
		
		phase->phase_info_gpu[gpu_index]->n_mono_types=phase->n_mono_types;
		phase->phase_info_gpu[gpu_index]->num_all_beads=phase->num_all_beads;
		phase->phase_info_gpu[gpu_index]->num_all_beads_per_node=phase->num_all_beads_per_node;
		phase->phase_info_gpu[gpu_index]->num_all_beads_per_gpu=phase->num_all_beads_per_gpu;

		phase->phase_info_gpu[gpu_index]->MaxThreadDensity=phase->MaxThreadDensity;
		phase->phase_info_gpu[gpu_index]->MaxThreadPolymer=phase->MaxThreadPolymer;
		
		

		checkCudaErrors(cudaMallocManaged((void**)&(phase->phase_info_gpu[gpu_index]->n_polymer_type),sizeof(unsigned int)*phase->polymer_type_number));
		checkCudaErrors(cudaMallocManaged((void**)&(phase->phase_info_gpu[gpu_index]->n_polymers_type_per_node),sizeof(unsigned int)*phase->polymer_type_number));
		checkCudaErrors(cudaMallocManaged((void**)&(phase->phase_info_gpu[gpu_index]->n_polymers_type_per_gpu),sizeof(unsigned int)*phase->polymer_type_number));

		for(int i=0;i<phase->polymer_type_number;i++){
			phase->phase_info_gpu[gpu_index]->n_polymer_type[i]=phase->n_polymer_type[i];
			phase->phase_info_gpu[gpu_index]->n_polymers_type_per_node[i]=phase->n_polymers_type_per_node[i];
			phase->phase_info_gpu[gpu_index]->n_polymers_type_per_gpu[i]=phase->n_polymers_type_per_gpu[i];

			

		}

		checkCudaErrors(cudaMallocManaged((void**)&(phase->phase_info_gpu[gpu_index]->num_bead_type),sizeof(unsigned int)*phase->n_mono_types));

		checkCudaErrors(cudaMallocManaged((void**)&(phase->phase_info_gpu[gpu_index]->num_bead_type_per_node),sizeof(unsigned int)*phase->n_mono_types));

		checkCudaErrors(cudaMallocManaged((void**)&(phase->phase_info_gpu[gpu_index]->num_bead_type_per_gpu),sizeof(unsigned int)*phase->n_mono_types));

		checkCudaErrors(cudaMallocManaged((void**)&(phase->phase_info_gpu[gpu_index]->num_bead_polymer_type),sizeof(unsigned int)*phase->polymer_type_number));

		checkCudaErrors(cudaMallocManaged((void**)&(phase->phase_info_gpu[gpu_index]->num_bead_polymer_type_per_node),sizeof(unsigned int)*phase->polymer_type_number));

		checkCudaErrors(cudaMallocManaged((void**)&(phase->phase_info_gpu[gpu_index]->num_bead_polymer_type_per_gpu),sizeof(unsigned int)*phase->polymer_type_number));

		checkCudaErrors(cudaMallocManaged((void**)&(phase->phase_info_gpu[gpu_index]->polymer_basis_gpu),sizeof(unsigned int)*(phase->polymer_type_number+1)));
	
		checkCudaErrors(cudaMallocManaged((void**)&(phase->phase_info_gpu[gpu_index]->monomer_poly_basis_gpu),sizeof(unsigned int)*(phase->polymer_type_number+1)));
		
		phase->phase_info_gpu[gpu_index]->polymer_basis_gpu[0]=0;

		phase->phase_info_gpu[gpu_index]->monomer_poly_basis_gpu[0]=0;

		for(int i=0;i<phase->polymer_type_number;i++){

			phase->phase_info_gpu[gpu_index]->num_bead_polymer_type[i]=phase->num_bead_polymer_type[i];
			
			phase->phase_info_gpu[gpu_index]->num_bead_polymer_type_per_node[i]=phase->num_bead_polymer_type_per_node[i];

			phase->phase_info_gpu[gpu_index]->num_bead_polymer_type_per_gpu[i]=phase->num_bead_polymer_type_per_gpu[i];
			
			phase->phase_info_gpu[gpu_index]->monomer_poly_basis_gpu[i+1]=phase->phase_info_gpu[gpu_index]->monomer_poly_basis_gpu[i]+phase->num_bead_polymer_type_per_gpu[i];

			phase->phase_info_gpu[gpu_index]->polymer_basis_gpu[i+1]=phase->phase_info_gpu[gpu_index]->polymer_basis_gpu[i]+phase->n_polymers_type_per_gpu[i];

		}

		for(int polymer_index=0; polymer_index<phase->polymer_type_number;polymer_index++){
			checkCudaErrors(cudaMallocManaged(&phase->poly_arch[gpu_index][polymer_index].mono_type_length,phase->n_mono_types));
	
			for(int i=0;i<phase->n_mono_types;i++) phase->poly_arch[gpu_index][polymer_index].mono_type_length[i]=0;

			for(int i=0;i<phase->poly_arch[gpu_index][polymer_index].poly_length;i++)  phase->poly_arch[gpu_index][polymer_index].mono_type_length[phase->poly_arch[gpu_index][polymer_index].Monotype[i]]++;


		}

		for(int i=0;i<phase->n_mono_types;i++){
			phase->num_bead_type[i]=0;
			phase->num_bead_type_per_node[i]=0;
			phase->num_bead_type_per_gpu[i]=0;
			for(int j=0;j<phase->polymer_type_number;j++){

				phase->num_bead_type[i]+=phase->poly_arch[gpu_index][j].mono_type_length[i]*phase->n_polymer_type[j];
				phase->num_bead_type_per_node[i]+=phase->poly_arch[gpu_index][j].mono_type_length[i]*phase->n_polymers_type_per_node[j];
				phase->num_bead_type_per_gpu[i]+=phase->poly_arch[gpu_index][j].mono_type_length[i]*phase->n_polymers_type_per_gpu[j];
			}

			phase->phase_info_gpu[gpu_index]->num_bead_type[i]=phase->num_bead_type[i];
			phase->phase_info_gpu[gpu_index]->num_bead_type_per_node[i]=phase->num_bead_type_per_node[i];
			phase->phase_info_gpu[gpu_index]->num_bead_type_per_gpu[i]=phase->num_bead_type_per_gpu[i];

		}
		//printf("%d %d %d %d %d\n",phase->num_bead_type_per_node[0],phase->num_bead_type_per_node[1],phase->n_polymers_type_per_node[0],phase->n_polymers_type_per_node[1],phase->n_polymers_type_per_node[2]);
		
		phase->phase_info_gpu[gpu_index]->reference_Nbeads=phase->reference_Nbeads;

		phase->phase_info_gpu[gpu_index]->inverse_refbeads=1/(double)phase->reference_Nbeads;

		phase->phase_info_gpu[gpu_index]->harmonic_normb=phase->harmonic_normb;

		checkCudaErrors(cudaMallocManaged((void**)&(phase->phase_info_gpu[gpu_index]->xn),sizeof(double)*phase->n_mono_types*phase->n_mono_types));

		checkCudaErrors(cudaMallocManaged((void**)&(phase->phase_info_gpu[gpu_index]->field_scaling_type),sizeof(double)*phase->n_mono_types));

		for(int t_type=0;t_type<phase->n_mono_types;t_type++)
			for(int s_type=0;s_type<phase->n_mono_types;s_type++)
				phase->phase_info_gpu[gpu_index]->xn[t_type+phase->n_mono_types*s_type]=phase->xn[t_type][s_type];
			
	}
	
	 // Max safe move distance
	phase->max_safe_jump = phase->Lx/phase->nx < phase->Ly / phase->ny ? phase->Lx/phase->nx : phase->Ly / phase->ny;
	phase->max_safe_jump = phase->max_safe_jump < phase->Lz / phase->nz ? phase->max_safe_jump : phase->Lz/phase->nz;
	phase->max_safe_jump *= 0.95;

	
	int Nx,Ny,Nz;
	
	factor_decompose(gpu_info,phase->n_cells/phase->MaxThreadDensity,&Nx,&Ny,&Nz);
	gpu_info->gridNx=Nx;
	gpu_info->gridNy=Ny*Nz;
	gpu_info->gridNz=phase->n_mono_types;
	
	phase->gridNx=gpu_info->gridNx;
	phase->gridNy=gpu_info->gridNy;
	phase->gridNz=gpu_info->gridNz;

	
	
	factor_decompose(gpu_info,phase->n_polymers_per_gpu/phase->MaxThreadPolymer,&Nx,&Ny,&Nz);
	gpu_info->polymerNx=Nx;
	gpu_info->polymerNy=Ny;
	gpu_info->polymerNz=Nz;
	
	phase->polymerNx=gpu_info->polymerNx;
	phase->polymerNy=gpu_info->polymerNy;
	phase->polymerNz=gpu_info->polymerNz;

	//printf("%d %d %d %d\n",phase->MaxThreadDensity,gpu_info->gridNx,gpu_info->gridNy,gpu_info->gridNz);
	//printf("%d %d %d %d\n",phase->MaxThreadPolymer,gpu_info->polymerNx,gpu_info->polymerNy,gpu_info->polymerNz);
	
	

	fclose(dp);


//---------------------------test--------------------------
/*
	printf("D0 =     %lg; LY =    %lg; LZ =    %lg;\n",p->Lx,p->Ly,p->Lz);
	printf("N = %d %d %d \n",p->nx,p->ny,p->nz);

	printf("time =  %d\n",p->time);
	
	printf("dt*N/xi = %lg\n",p->dtNxi);
	printf("kT = %lg\n",p->kT);
	printf("TCHECK = %d\n",p->Tcheck);
	printf("TWRITE = %d\n",p->Twrite);


	for(int j=0;j<p->n_mono_types;j++){

	
		for(int i=0;i<p->n_mono_types;i++) {
			printf("%g ",p->xn[i][j]);
		}
		printf("\n");
	}
*/


}
extern int initialize_structure_GPU(GPU_info *gpu_info,Phase *phase){

	phase->pos.resize(gpu_info->GPU_N);
	phase->fields_unified.resize(gpu_info->GPU_N);
	phase->fields_32.resize(gpu_info->GPU_N);
	phase->omega_field_unified.resize(gpu_info->GPU_N);
	phase->temp_average_field_unified.resize(gpu_info->GPU_N);
	phase->average_field_unified.resize(gpu_info->GPU_N);	
	
	//size_t available, total;
	
	
	for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){// gpu_info->GPU_N
		
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	
		
		checkCudaErrors(cudaMallocManaged(&(phase->pos[gpu_index]),sizeof(float)*phase->num_all_beads_per_gpu*3));//position of monomer
		
		checkCudaErrors(cudaMallocManaged(&(gpu_info->state[gpu_index]),sizeof(curandStatePhilox4_32_10_t)*phase->n_polymers_per_gpu));
		

		checkCudaErrors(cudaMallocManaged(&(phase->fields_unified[gpu_index]),phase->n_cells*phase->n_mono_types*sizeof(int)));

		
		checkCudaErrors(cudaMallocManaged(&(phase->fields_32[gpu_index]),phase->n_cells*phase->n_mono_types*sizeof(uint32_t)));

		
		checkCudaErrors(cudaMallocManaged(&(phase->omega_field_unified[gpu_index]),phase->n_mono_types*phase->n_cells*sizeof(double)));

		
		checkCudaErrors(cudaMallocManaged(&(phase->temp_average_field_unified[gpu_index]),phase->n_mono_types*phase->n_mono_types*sizeof(int)));

		
		checkCudaErrors(cudaMallocManaged(&(phase->average_field_unified[gpu_index]),phase->n_cells*phase->n_mono_types*sizeof(int)));	

	}//!< loop for gpu_index gpu device

	//printf("monomer type number is %d cell %d\n",phase->n_mono_types,phase->n_cells);

	phase->field_scaling_type = (double *) malloc(phase->n_mono_types * sizeof(double));
    	if (phase->field_scaling_type == NULL) {
		fprintf(stderr, "ERROR: Malloc %s:%d\n", __FILE__, __LINE__);
		return -1;
    	}
		
	long ncells = phase->n_cells;
	
	for (unsigned int i = 0; i < phase->n_mono_types; i++)
		phase->field_scaling_type[i] =(ncells / ((double) phase->num_all_beads));

	for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){// gpu_info->GPU_N

	
		for(int i=0;i<phase->n_mono_types;i++)

		phase->phase_info_gpu[gpu_index]->field_scaling_type[i]=phase->field_scaling_type[i];

	}

	phase->R = (double*) malloc( phase->n_mono_types * sizeof(double));
	phase->A  = (double*)malloc( phase->n_mono_types * sizeof(double));
	
        if(phase->R == NULL){
                   fprintf(stderr, "ERROR: By malloc TT800 , %s %d ",__FILE__, __LINE__ );
                    return -1;
         }
	
	for (unsigned int i = 0; i < phase->n_mono_types; i++){
		//! \todo kBT required.
			//p->A[i] 
		phase->A[i] =phase->dtNxi/ phase->reference_Nbeads;
		phase->R[i] = sqrt( phase->A[i]* 2);
	}
	
	phase->n_accepts = 0;
	phase->n_moves = 0;

	for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){// gpu_info->GPU_N
		
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	

		checkCudaErrors(cudaMallocManaged(&(phase->phase_info_gpu[gpu_index]->A),phase->n_mono_types));

		checkCudaErrors(cudaMallocManaged(&(phase->phase_info_gpu[gpu_index]->R),phase->n_mono_types));

		//phase->phase_info_gpu[gpu_index]->omega_field_unified=phase->omega_field_unified[gpu_index];
		//phase->phase_info_gpu[gpu_index]->area51=phase->area51[gpu_index];

		phase->phase_info_gpu[gpu_index]->max_safe_jump=phase->max_safe_jump;

		

		for (unsigned int i = 0; i < phase->n_mono_types; i++){
		//! \todo kBT required.
			//p->A[i] 
			phase->phase_info_gpu[gpu_index]->A[i]=phase->A[i];
			phase->phase_info_gpu[gpu_index]->R[i]=phase->R[i];
		}


	}// end loop gpu

    // initialize inverse simulation cell parameters
	phase->iLx = 1.0/phase->Lx;
	phase->iLy = 1.0/phase->Ly;
   	phase->iLz = 1.0/phase->Lz;

	phase->dx=phase->Lx/phase->nx;
	phase->dy=phase->Ly/phase->ny;
	phase->dz=phase->Lz/phase->nz;

	//phase->sets = NULL; // Default init of the sets
   	phase->max_set_members = 0;
	
	return 1;	
	
	
}// end routine

extern int initialize_random_generator(MPI_info *mpi_info,GPU_info *gpu_info,Phase *phase){

	unsigned int fixed_seed;
	
	int fix=0;	
	
	if(fix==0)

		fixed_seed = time(NULL);
    	else
		fixed_seed = 1;

	

	
	
	
	
	for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){

		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	

		phase->phase_info_gpu[gpu_index]->current_node=mpi_info->current_node;
		
		phase->phase_info_gpu[gpu_index]->total_nodes=mpi_info->total_nodes;
				
		dim3 grid(gpu_info->polymerNx,gpu_info->polymerNy,gpu_info->polymerNz);
		
			
		setup_curand<<<grid,phase->MaxThreadPolymer,0,gpu_info->stream[gpu_index]>>>(phase->phase_info_gpu[gpu_index],gpu_index, fixed_seed,gpu_info->state[gpu_index]);
		
		checkCudaErrors( cudaDeviceSynchronize());
	}
			
		

	
	
	return 0;
} 


extern void Generate_init_coord(MPI_info *mpi_info,GPU_info *gpu_info,Phase *phase){
	

	int read_file=phase->read_file;
	
	if(read_file==0){
	
		for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){

			checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	// 
			
			//size_t mem=sizeof(int)*1024;
			//printf("c %ld %d %d %d\n",phase->phase_info_gpu[gpu_index]->n_cells,phase->phase_info_gpu[gpu_index]->nx,phase->phase_info_gpu[gpu_index]->ny,phase->phase_info_gpu[gpu_index]->nz);
			
			dim3 grid(gpu_info->polymerNx,gpu_info->polymerNy,gpu_info->polymerNz);
			//printf("%d %d %d %d\n",grid.x,grid.y,grid.z,phase->MaxThreadPolymer);
			initialize_coord<<<grid,phase->MaxThreadPolymer,0,gpu_info->stream[gpu_index]>>>(phase->pos[gpu_index],phase->phase_info_gpu[gpu_index],phase->poly_arch[gpu_index],gpu_info->state[gpu_index]);
			
		
			//checkCudaErrors( cudaDeviceSynchronize());
			//printf("dx %g dy %g  dz  %g\n",phase->phase_info_gpu[gpu_index]->dx,phase->phase_info_gpu[gpu_index]->dy,phase->phase_info_gpu[gpu_index]->dz);
		}

		for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){

			checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	// 
			checkCudaErrors( cudaDeviceSynchronize());

		}


	}
	else if(read_file==1){

		FILE *dp;
		

		for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){

			checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	// 
			
			//size_t mem=sizeof(int)*1024;
			//printf("c %ld %d %d %d\n",phase->phase_info_gpu[gpu_index]->n_cells,phase->phase_info_gpu[gpu_index]->nx,phase->phase_info_gpu[gpu_index]->ny,phase->phase_info_gpu[gpu_index]->nz);
			
			dim3 grid(gpu_info->polymerNx,gpu_info->polymerNy,gpu_info->polymerNz);
			//printf("%d %d %d %d\n",grid.x,grid.y,grid.z,phase->MaxThreadPolymer);
			initialize_coord<<<grid,phase->MaxThreadPolymer,0,gpu_info->stream[gpu_index]>>>(phase->pos[gpu_index],phase->phase_info_gpu[gpu_index],phase->poly_arch[gpu_index],gpu_info->state[gpu_index]);
			
		
			checkCudaErrors( cudaDeviceSynchronize());
			//printf("dx %g dy %g  dz  %g\n",phase->phase_info_gpu[gpu_index]->dx,phase->phase_info_gpu[gpu_index]->dy,phase->phase_info_gpu[gpu_index]->dz);
		}
		
		dp=fopen("coord.dat","r");
		if(dp==NULL) {printf("coord.dat did not exit ^_^, program smartly change config to generate a random coord.\n");return;}
		
			
		int cpu_index=mpi_info->current_node;
		int move_to=cpu_index*phase->num_all_beads_per_node;//cpu_index*phase->num_all_beads_per_node+gpu_index*phase->num_all_beads_per_gpu;
		double a,b,c;
		int d;
		for(int i=0;i<move_to;i++) fscanf(dp,"%lg %lg %lg\n",&a,&b,&c,&d);
		
			
			
			
		for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){

			
			
			
			for(int i=0;i<phase->num_all_beads_per_gpu;i++){
				fscanf(dp,"%lg %lg %lg\n",&a,&b,&c,&d);
				phase->pos[gpu_index][i*3]=a;
				phase->pos[gpu_index][i*3+1]=b;
				phase->pos[gpu_index][i*3+2]=c;
				//if(gpu_index==1&&i%32000==0) printf("%g %g %g\n",a,b,c);
				
			
			}
		}//end loop gpu
			
		fclose(dp);
		

		
		
		

	}// end read_file

	

}
extern void init_all_config(GPU_info *gpu_info,Phase *phase, MPI_info *mpi_info,int argc, char **argv){

	init_scmf(phase,gpu_info,argc, argv);
	
	init_cuda(mpi_info,gpu_info,0);
	
	Read_polymer_config(mpi_info,gpu_info,phase);
	
	initialize_values(gpu_info,phase);
	
	initialize_structure_GPU(gpu_info,phase);

	initialize_random_generator(mpi_info,gpu_info,phase);

	Generate_init_coord(mpi_info, gpu_info,phase);
	//printf("density %d %d %d %d\n",phase->gridNx,phase->gridNy,phase->gridNz,phase->MaxThreadDensity);
	//printf("polymer %d %d %d %d\n",phase->polymerNx,phase->polymerNy,phase->polymerNz,phase->MaxThreadPolymer);
	
	
}

