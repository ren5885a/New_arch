#include "device_function.cuh"
// This is a interface programm(call from Host and run on Device). 
// It initializes the random generator(curandStatePhilox4_32_10_t) on a random state array (state). 
// ---------------------------------------------------------------------
// Input: phase_info_gpu which contains all the phase information.
//        gpu_index is the index of GPU on one computer node.
//        seed is the first seed of random number generator.
// Output:state
// Method: Number of thread should be equal to the length of the state. Example: for an array of state with 4000 conponents, call(4,1000).
// Notation: Preassume maximal 8 GPU per node.

__global__ void setup_curand(Phase_info_gpu *phase_info_gpu,int gpu_index,unsigned int seed,curandStatePhilox4_32_10_t *state){
	
	int id=(blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z)*blockDim.x+threadIdx.x;
		
	int current_node=phase_info_gpu->current_node;
	
	long index=id+(current_node*8+gpu_index)*phase_info_gpu->n_polymers_per_gpu;
	
	
	curand_init(seed, index, 0, &state[id]);
	
	__syncthreads();
}


// AtomicAdd is download from Nvidia Co. website. As an atomic operation for double type is not contained by defalt.
// This program is used for density and field calculation on Multiple GPUs. 

__device__ double atomicAdd(double* address, double val) { 
	
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	
	do { 
		assumed = old; 
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) 
	} while (assumed != old); 

	return __longlong_as_double(old); 

}
 __device__ float atomicAddFloat(float* address, float value){

      float old = value;  

      float ret=atomicExch(address, 0.0f);

      float new_old=ret+old;



      while ((old = atomicExch(address, new_old))!=0.0f)

      {

    	new_old = atomicExch(address, 0.0f);

    	new_old += old;

      }
	return 0;
}

// This is a device function.
// It transform the index of polymer in the GPU to the type of polymer and the index in this type of polymer.  
//------------------------------------------------------------------------------
// Input: Id is the index of polymer in the GPU
//        poly_arch structure information of all type of polymer.
// Output: dim3.x index of polymer type dim3.y polymer index in his type. 

__device__ dim3 polymer_number_to_index(int id,Phase_info_gpu *phase_info_gpu,Poly_arch *poly_arch){

		
	dim3 pos_index;

	
	//printf("%d %d %d %d\n",phase_info_gpu->polymer_basis_gpu[0],phase_info_gpu->polymer_basis_gpu[1],phase_info_gpu->polymer_basis_gpu[2],phase_info_gpu->polymer_basis_gpu[3]);
	//printf("%d %d %d %d\n",phase_info_gpu->monomer_poly_basis_gpu[0],phase_info_gpu->monomer_poly_basis_gpu[1],phase_info_gpu->monomer_poly_basis_gpu[2],phase_info_gpu->monomer_poly_basis_gpu[3]);
	
	for(int i=0;i<phase_info_gpu->polymer_type_number;i++) 
		if(id<phase_info_gpu->polymer_basis_gpu[i+1]) {
			pos_index.x=i;
			pos_index.y=id-phase_info_gpu->polymer_basis_gpu[i];
			
			break;
			

		}
		
	
	return pos_index;
}

// This is a device function.
// Postion of monomers is stored in a 1D array pos, it can only be obtained by knowing its pos_index in this array.
// pos_index is returned by given polymer_index and mono_index
//----------------------------------------------------------------------
//Input:
//      arch_index: polymer type. 
//      polymer_index: index of polymer in its type.
//      mono_index: index of monomer in this polymer.
//Output: pos_index position of monomer is   x=pos[pos_index*3] y=pos[pos_index*3+1]  z=pos[pos_index*3+2] 
__device__ int polymer_index_to_pos(int arch_index,int polymer_index,int mono_index,Phase_info_gpu *phase_info_gpu,Poly_arch *poly_arch){

	

	int pos_index=phase_info_gpu->monomer_poly_basis_gpu[arch_index]+polymer_index*poly_arch[arch_index].poly_length+mono_index;

	return pos_index;



}

// This is a device function.
// Given the index of polymer in the GPU and monomer index return the position of monomer in pos.
// Input: id : index of polymer in the GPU
//        mono_index: monomer index 
// Output: pos_index in pos x=pos[pos_index*3] y=pos[pos_index*3+1]  z=pos[pos_index*3+2] 
__device__ int polymer_number_to_pos(int id,int mono_index,Phase_info_gpu *phase_info_gpu,Poly_arch *poly_arch){

	dim3 index;

	index=polymer_number_to_index(id,phase_info_gpu,poly_arch);
	//printf("%d %d\n",index.x,index.y);
	//printf("%d %d %d\n",phase_info_gpu->polymer_basis_gpu[0],phase_info_gpu->polymer_basis_gpu[1],phase_info_gpu->polymer_basis_gpu[2]);
	int pos_index=polymer_index_to_pos(index.x,index.y,mono_index,phase_info_gpu,poly_arch);

	

	return pos_index;
}

// Interface programm to initialize coordinates
// Input: pos : coordinates of all monomer
// Method : (polymer_number_gpu/THREAD_PER_BLOCK,THREAD_PER_BLOCK)
//Maximal polymer length is 1024

__global__ void initialize_coord(float *pos,Phase_info_gpu *phase_info_gpu,Poly_arch *poly_arch,curandStatePhilox4_32_10_t *state){

	
	int id=(blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z)*blockDim.x+threadIdx.x;

	
	
	 
	dim3 arch_poly_index=polymer_number_to_index(id,phase_info_gpu,poly_arch);
	
	int pos_index=polymer_number_to_pos(id,0,phase_info_gpu,poly_arch);
	

	
	int occupy[100];
	

	for(int i=0;i<poly_arch[arch_poly_index.x].poly_length;i++)	occupy[i]=0;

	if(pos_index>phase_info_gpu->num_all_beads_per_gpu) printf("over array bond %d %d\n",id,pos_index);
	if(id>phase_info_gpu->n_polymers_per_gpu) printf("over array bond %d %d\n",id,pos_index);
	
	
	pos[pos_index*3]=curand_uniform(&state[id])*phase_info_gpu->Lx;
	pos[pos_index*3+1]=curand_uniform(&state[id])*phase_info_gpu->Ly;
	pos[pos_index*3+2]=curand_uniform(&state[id])*phase_info_gpu->Lz;

	occupy[0]=1;

	double reference_Nbeads=poly_arch[arch_poly_index.x].poly_length;

	double harmonic_spring_Cste =1.0 / sqrt(3.0 * (reference_Nbeads - 1.0));

	//if(id==0) printf("%g %g\n",reference_Nbeads,harmonic_spring_Cste);

	for(int i=0;i<poly_arch[arch_poly_index.x].poly_length;i++){

		for(int neigh_index=0;neigh_index<poly_arch[arch_poly_index.x].neigh_num[i];neigh_index++){
		
			int index=poly_arch[arch_poly_index.x].conection_list[i][neigh_index];

			if(occupy[index]==0){
				
				double4 position=curand_normal4_double(&state[id]);
				
				pos[(pos_index+index)*3]=position.x*harmonic_spring_Cste+pos[(pos_index+i)*3];
				pos[(pos_index+index)*3+1]=position.y*harmonic_spring_Cste+pos[(pos_index+i)*3+1];
				pos[(pos_index+index)*3+2]=position.z*harmonic_spring_Cste+pos[(pos_index+i)*3+2];

				//if(id==0) printf("%d %d %g %g %g\n",index,i,pos[(pos_index+index)*3],pos[(pos_index+index)*3+1],pos[(pos_index+index)*3+2]);

				occupy[index]=1;
		
			}//end if
			

		}//end for neigh


	}//end for i

	

	
	

__syncthreads();
}
/*
	printf("%d %d %d\n",phase_info_gpu->polymer_basis_gpu[0],phase_info_gpu->polymer_basis_gpu[1],phase_info_gpu->polymer_basis_gpu[2]);


	
	printf("n_polymers=%d n_polymers_per_node=%d n_polymers_per_gpu %d\n",phase_info_gpu->n_polymers,phase_info_gpu->n_polymers_per_node,phase_info_gpu->n_polymers_per_gpu);

	printf("polymer_type_number %d\n",phase_info_gpu->polymer_type_number);
	for(int i=0;i<phase_info_gpu->polymer_type_number;i++)
	printf("n_polymer_type %d: %d ",i,phase_info_gpu->n_polymer_type[i]);
	printf("\n");
	for(int i=0;i<phase_info_gpu->polymer_type_number;i++)
	printf("n_polymers_type_per_node %d: %d ",i,phase_info_gpu->n_polymers_type_per_node[i]);
	printf("\n");
	for(int i=0;i<phase_info_gpu->polymer_type_number;i++)
	printf("n_polymers_type_per_gpu %d: %d ",i,phase_info_gpu->n_polymers_type_per_gpu[i]);
	printf("\n");

	for(int i=0;i<phase_info_gpu->polymer_type_number;i++)
	printf("num_bead_polymer_type %d: %d ",i,phase_info_gpu->num_bead_polymer_type[i]);
	printf("\n");
	for(int i=0;i<phase_info_gpu->polymer_type_number;i++)
	printf("num_bead_polymer_type_per_node %d: %d ",i,phase_info_gpu->num_bead_polymer_type_per_node[i]);
	printf("\n");
	for(int i=0;i<phase_info_gpu->polymer_type_number;i++)
	printf("num_bead_polymer_type_per_gpu %d: %d ",i,phase_info_gpu->num_bead_polymer_type_per_gpu[i]);
	printf("\n");


	printf("n_mono_types=%d num_all_beads=%d num_all_beads_per_node %d\n",phase_info_gpu->n_mono_types,phase_info_gpu->num_all_beads,phase_info_gpu->num_all_beads_per_node);
	for(int i=0;i<phase_info_gpu->n_mono_types;i++)
	printf("num_bead_type %d: %d ",i,phase_info_gpu->num_bead_type[i]);
	printf("\n");
	for(int i=0;i<phase_info_gpu->n_mono_types;i++)
	printf("num_bead_type_per_node %d: %d ",i,phase_info_gpu->num_bead_type_per_node[i]);
	printf("\n");

	for(int i=0;i<phase_info_gpu->n_mono_types;i++)
	printf("num_bead_type_per_gpu %d: %d ",i,phase_info_gpu->num_bead_type_per_gpu[i]);
	printf("\n");


	for(int i=0;i<phase_info_gpu->n_mono_types;i++)
		printf("field_scaling_type %g ",phase_info_gpu->field_scaling_type[i]);
	printf("\n");

	printf("inverse_refbeads %g reference_Nbeads %d harmonic_normb %g\n",phase_info_gpu->inverse_refbeads,phase_info_gpu->reference_Nbeads,phase_info_gpu->harmonic_normb);

	for(int i=0;i<phase_info_gpu->n_mono_types;i++)
		printf("A[%d] %g R[%d] %g ",i,phase_info_gpu->A[i],i,phase_info_gpu->R[i]);
	printf("\n");

	printf("Lx %g Ly %g Lz %g iLx %g iLy %g iLz %g\n",phase_info_gpu->Lx,phase_info_gpu->Ly,phase_info_gpu->Lz,phase_info_gpu->iLx,phase_info_gpu->iLy,phase_info_gpu->iLz);

	
	printf("ncell %ld nx %d ny %d nz %d\n",phase_info_gpu->n_cells,phase_info_gpu->nx,phase_info_gpu->ny,phase_info_gpu->nz);

*/

// This is a device programm.
// coordinate of monomer is transformed to cell index rx ry and rz.

__device__ void coord_to_cell_coordinate(Phase_info_gpu *phase_info_gpu, const double rx, const double ry, const double rz, int * x, int * y, int * z){

  double px, py, pz;

  //\todo Optimization: store inverse box length for instruction optimization
	

  // Fold coordinate back into the box
  px = rx - phase_info_gpu->Lx * (int) (rx / phase_info_gpu->Lx);
  py = ry - phase_info_gpu->Ly * (int) (ry / phase_info_gpu->Ly);
  pz = rz -phase_info_gpu->Lz * (int) (rz /phase_info_gpu->Lz);

  // Assure correct symmetry at coordinate = 0
  if (px < 0 ) px =phase_info_gpu->Lx +px;
  if (py < 0 ) py = phase_info_gpu->Ly +py;
  if (pz < 0 ) pz = phase_info_gpu->Lz +pz;

  // Calculate index
  *x = (int)( px /phase_info_gpu->Lx * phase_info_gpu->nx);
  *y = (int)( py /phase_info_gpu->Ly * phase_info_gpu->ny);
  *z = (int)( pz /phase_info_gpu->Lz *phase_info_gpu->nz);

}

// from cell index transformed to cell+type*cell_number
__device__ unsigned int cell_to_index_unified(Phase_info_gpu *phase_info_gpu,const unsigned int cell,const unsigned int rtype)
    {
//Unified data layout [type][x][y][z]
    return cell + rtype * phase_info_gpu->n_cells;
    }
/*! calculate the field array index from 3 spatial coordinates */
__device__  unsigned int coord_to_index( Phase_info_gpu *phase_info_gpu, const double rx,
			    const double ry, const double rz)
    {

    int x, y, z;

    coord_to_cell_coordinate(phase_info_gpu, rx, ry, rz, &x, &y, &z);

    return cell_coordinate_to_index(phase_info_gpu, x, y, z);
    }


__device__ unsigned int coord_to_index_unified_high(const double rx,const double ry, const double rz, const unsigned int rtype,Phase_info_gpu *phase_info_gpu){

		int xi,yi,zi;

		if(rx>=0){
			
			int Lx=(int)(rx/phase_info_gpu->Lx);
			xi=(int)((rx-phase_info_gpu->Lx*Lx)/phase_info_gpu->dx);
			xi=xi%phase_info_gpu->nx;
		}
		else if(rx<0){

			int Lx=(int)(-rx/phase_info_gpu->Lx)+1;
			xi=(int)((Lx*phase_info_gpu->Lx+rx)/phase_info_gpu->dx);
			xi=xi%phase_info_gpu->nx;

		}
		if(ry>=0){
			
			int Ly=(int)(ry/phase_info_gpu->Ly);
			yi=(int)((ry-phase_info_gpu->Ly*Ly)/phase_info_gpu->dy);
			yi=yi%phase_info_gpu->ny;
			
		}
		else if(ry<0){

			int Ly=(int)(-ry/phase_info_gpu->Ly)+1;
			yi=(int)((Ly*phase_info_gpu->Ly+ry)/phase_info_gpu->dy);
			yi=yi%phase_info_gpu->ny;

		}
		if(rz>=0){
			
			int Lz=(int)(rz/phase_info_gpu->Lz);
			zi=(int)((rz-phase_info_gpu->Lz*Lz)/phase_info_gpu->dz);
			zi=zi%phase_info_gpu->nz;
			
		}
		else if(rz<0){

			int Lz=(int)(-rz/phase_info_gpu->Lz)+1;
			zi=(int)((Lz*phase_info_gpu->Lz+rz)/phase_info_gpu->dz);
			zi=zi%phase_info_gpu->nz;

		}
		//if(threadIdx.x==0&&blockIdx.x==0&&blockIdx.y==0&&blockIdx.z==0) printf("x %d y%d z %d \n",xi,yi,zi);
		return (rtype*phase_info_gpu->n_cells+xi+yi*phase_info_gpu->nx+zi*phase_info_gpu->nx*phase_info_gpu->ny);
}
/*! calculate the field array index from 3 spatial coordinates for the unified fields/external fields */
__device__ unsigned int coord_to_index_unified( const double rx,const double ry, const double rz, const unsigned int rtype,Phase_info_gpu *phase_info_gpu){

	int x, y, z;

	coord_to_cell_coordinate( phase_info_gpu,rx, ry, rz, &x, &y, &z);
    	const unsigned int cell =cell_coordinate_to_index(phase_info_gpu, x, y, z);
	
	return cell_to_index_unified(phase_info_gpu,cell,rtype);
    }
	

__global__ void omega_field_update(int *density,double *omega_field,Phase_info_gpu *phase_info_gpu){

	long cell=(blockIdx.x+gridDim.x*blockIdx.y)*blockDim.x+threadIdx.x;
		
	int T_types=blockIdx.z;

	omega_field[cell + T_types*phase_info_gpu->n_cells]=0;

	for (unsigned int S_types = 0; S_types < phase_info_gpu->n_mono_types;S_types++){

		omega_field[cell + T_types*phase_info_gpu->n_cells] += density[cell+S_types*phase_info_gpu->n_cells]*phase_info_gpu->field_scaling_type[T_types];

	}
	omega_field[cell + T_types*phase_info_gpu->n_cells]+=-1;

	
	omega_field[cell + T_types*phase_info_gpu->n_cells]=phase_info_gpu->inverse_refbeads*(phase_info_gpu->xn[T_types+phase_info_gpu->n_mono_types*T_types] * omega_field[cell + T_types*phase_info_gpu->n_cells]);

	
	
	for (unsigned int S_types = 0; S_types < phase_info_gpu->n_mono_types;S_types++){
		
		if(T_types!=S_types){
		
			double dnorm = -0.5 *  phase_info_gpu->xn[T_types+phase_info_gpu->n_mono_types*S_types];
		
			double interaction = dnorm *phase_info_gpu->inverse_refbeads *
		    			(phase_info_gpu->field_scaling_type[T_types]*density[cell+T_types*phase_info_gpu->n_cells] - phase_info_gpu->field_scaling_type[S_types]*density[cell+S_types*phase_info_gpu->n_cells]); /*Added the rescaling cause p->fields are short now*/
					omega_field[cell+T_types*phase_info_gpu->n_cells] += interaction;
		
		}
					

	}

	

__syncthreads();	

}
__global__ void reduce_field_int(int *density_dst,int *density_res,long size,int GPU_N){

	int cell=(blockIdx.x+gridDim.x*blockIdx.y)*blockDim.x+threadIdx.x;

	for(int i=1;i<GPU_N;i++){
		
		density_dst[cell+blockIdx.z*size]+=density_res[cell+blockIdx.z*size+i*size*gridDim.z];

	}


__syncthreads();

}

__global__ void reduce_field_double(double *density_dst,double *density_res,long size,int GPU_N){

	int cell=(blockIdx.x+gridDim.x*blockIdx.y)*blockDim.x+threadIdx.x;

	

	for(int i=1;i<GPU_N;i++){
		
		density_dst[cell]+=density_res[cell+blockIdx.z*size];

	}


__syncthreads();

}
__device__ bool som_accept(curandStatePhilox4_32_10_t * rng,   double delta_energy)
{
  //! \todo kBT reqired
    const double p_acc = exp(-1.0 * delta_energy );

    if ((p_acc > 1) || (p_acc > curand_uniform(rng))) {	//Use lazy eval.
	return true;
    } else {
	return false;
    }
}
//type is monomer typer
// from(x2,y2,z2) move to (x1,y1,z1)
__device__ void update_density_local(float x1,float y1,float z1,float x2,float y2,float z2,int type, int *delta_density,Phase_info_gpu *phase_info_gpu){

	int add=coord_to_index_unified( x1,y1, z1, type,phase_info_gpu);

	int minus=coord_to_index_unified( x2,y2, z2, type,phase_info_gpu);
	
	
	atomicAdd(&delta_density[add], 1);
	atomicAdd(&delta_density[minus], -1);

__syncthreads();

}

__global__ void init_array(int *density,int size){

	long grid_index=(blockIdx.x+gridDim.x*blockIdx.y)*blockDim.x+threadIdx.x;
		
	int monomer_type=blockIdx.z;

	density[grid_index+size*monomer_type]=0;
}
__global__ void coord_to_density(float *pos,int *density,Phase_info_gpu *phase_info_gpu,Poly_arch *poly_arch){

	int polymer_index=(blockIdx.x+gridDim.x*blockIdx.y+gridDim.x*gridDim.y*blockIdx.z)*blockDim.x+threadIdx.x;
		
	dim3 index=polymer_number_to_index(polymer_index,phase_info_gpu,poly_arch);;

	
	for(int monomer_index=0;monomer_index<poly_arch[index.x].poly_length;monomer_index++){

		int xyz=polymer_number_to_pos(polymer_index,monomer_index,phase_info_gpu,poly_arch);
		
		if(xyz>=phase_info_gpu->num_all_beads_per_gpu) {printf("over size at coord_to_density.\n");}
				

		const double x=pos[3*xyz];
		const double y=pos[3*xyz+1];
		const double z=pos[3*xyz+2];
		
 		const unsigned int mono_type=poly_arch[index.x].Monotype[monomer_index];
	
		
		int cell=coord_to_index_unified_high(x ,y, z, mono_type,phase_info_gpu);
		
		if(cell>phase_info_gpu->n_cells*phase_info_gpu->n_mono_types) printf("too large %g %g %g %d %d %d %d\n",x,y,z,xyz,cell,mono_type,phase_info_gpu->n_cells*phase_info_gpu->n_mono_types);
		atomicAdd(&density[cell], 1);
	}
	
__syncthreads();
	
} 
__device__ double calc_delta_bonded_energy_arr(float *pos,int poly_id,int mono_id,int poly_type,Phase_info_gpu *phase_info_gpu,Poly_arch *poly_arch,
				 const double dx,const double dy, const double dz){ 
	double delta_energy = 0;
	
	const unsigned int bond_type=0; //= get_bond_type(info);

	const double harm_norb=phase_info_gpu->harmonic_normb;

	//const int mono_type=poly_arch[poly_type].Monotype[mono_id];

	const int poly_index= polymer_number_to_pos(poly_id,0,phase_info_gpu,poly_arch);
	
	switch (bond_type) {
		
		case HARMONIC:{
			//Empty statement, because a statement after a label
			//has to come before any declaration
			
			for(int jbeads=0;jbeads<poly_arch[poly_type].neigh_num[mono_id];jbeads++){ //loop all bond around ibeads
				
				const int neigh=poly_arch[poly_type].conection_list[mono_id][jbeads];
				
				const int neigh_index=poly_index+neigh;
				const int index=poly_index+mono_id;

				const double old_rx =pos[index*3]-pos[neigh_index*3];//polymers[ipoly].beads[ibead].x -polymers[ipoly].beads[neigh_index].x;
				const double new_rx = old_rx + dx;
	
				const double old_ry =pos[index*3+1]-pos[neigh_index*3+1];
				const double new_ry = old_ry + dy;
	
				const double old_rz =pos[index*3+2]-pos[neigh_index*3+2];
				const double new_rz = old_rz + dz;

				const double old_r2 =old_rx * old_rx + old_ry * old_ry + old_rz * old_rz;
				const double new_r2 =new_rx * new_rx + new_ry * new_ry + new_rz * new_rz;
		
				delta_energy += harm_norb * (new_r2 - old_r2);

			}//!< end loop for jbeads
		
			break;
		}
      		case STIFF:

		break;
		
	};// end switch

   	
 
	return delta_energy;
}
__device__ double calc_delta_energy_arr(int poly_id,int mono_id,int poly_type,float *pos,Phase_info_gpu *phase_info_gpu,Poly_arch *poly_arch,double *omega_field_unified,double xold,double yold,double zold,
			 double dx, double dy,double dz,const unsigned int iwtype){

    unsigned int cellindex_old;
    unsigned int cellindex_new;
   
    double energy_old, energy_new;
    double energy;

   

    cellindex_old =coord_to_index_unified_high(xold, yold, zold, iwtype,phase_info_gpu);
		
    energy_old =omega_field_unified[cellindex_old];

    cellindex_new = coord_to_index_unified_high(xold+dx, yold+dy, zold+dz, iwtype,phase_info_gpu);
			
    // New non-bonded interaction
    energy_new = omega_field_unified[cellindex_new];

    const double delta_bonded_energy =calc_delta_bonded_energy_arr(pos,poly_id,mono_id,poly_type,phase_info_gpu,poly_arch,dx, dy, dz);
	
    // non-bonded energy + bonded energy
    energy = energy_new - energy_old;
    energy += delta_bonded_energy;
    return energy;
}
__device__ void add_bond_forces_arr(int poly_type,int mono_id,int poly_index,Poly_arch *poly_arch, Phase_info_gpu *phase_info_gpu,double harm, float *pos,
                     float x, float y, const float z,
                     float *fx, float *fy, float *fz){

	double v1x=0.0,v1y=0.0,v1z=0.0;
	
	unsigned int bond_type=0;
	
	
	switch (bond_type) {
	
		case HARMONIC:{
	  	//Empty statement, because a statement after a label
	  	//has to come before any declaration

			for(int jbeads=0;jbeads<poly_arch[poly_type].neigh_num[mono_id];jbeads++){ //loop all bond around ibeads

				const int neigh=poly_arch[poly_type].conection_list[mono_id][jbeads];

				const int neigh_index=(poly_index+neigh)*3;
				
				if(neigh_index>phase_info_gpu->num_all_beads_per_gpu*3)	printf("add_bond_force_arr over index %d %d\n",neigh_index,poly_index);	
				
				v1x += (pos[neigh_index ] - x)*2.0*harm;
	  			v1y += (pos[neigh_index+1] - y)*2.0*harm;
	  			v1z += (pos[neigh_index+2] - z)*2.0*harm;
				

			}// end for jbeads
			
			break;
		}
		case STIFF:

		break;
	}// end switch
	
      	
	
   
   	
   	*fx += v1x;
   	*fy += v1y;
    	*fz += v1z;
	
}


__device__ void trial_move_smc_arr(curandStatePhilox4_32_10_t *state,float *pos,int poly_id,int mono_id,int poly_type,int poly_index,Phase_info_gpu *phase_info_gpu,Poly_arch *poly_arch, float *dx, float *dy, float *dz, double *   smc_deltaE){

	float x,y,z;
	float fx,fy,fz;
	float nfx,nfy,nfz;
	float A,R;
		
	int index=polymer_number_to_pos(poly_id,mono_id,phase_info_gpu,poly_arch);

	if(index>=phase_info_gpu->num_all_beads_per_gpu) {
		printf("trial_move_smc_arr: block Nx %d Ny %d Nz %d thread: %d \n",threadIdx.x,threadIdx.y,threadIdx.z);
		
		  __threadfence();         // ensure store issued before trap
 		 asm("trap;");            // kill kernel with error

	}

	x=pos[index*3];
	y=pos[index*3+1];
	z=pos[index*3+2];
	
	int iwtype=poly_arch[poly_type].Monotype[mono_id];
	
    // R calculated from A according to: Rossky, Doll and Friedman, J.Chem.Phys 69(10)1978 
	//A=phase_info_gpu->A[iwtype];
	//R=phase_info_gpu->R[iwtype];
	A=phase_info_gpu->A[iwtype];
	R=phase_info_gpu->R[iwtype];
    // calculate forces in current position
	fx=0.0; fy=0.0;fz=0.0;
	
	//if(poly_id==39999) printf("poly_index %d\n",poly_index);
	
	add_bond_forces_arr(poly_type,mono_id,poly_index,poly_arch, phase_info_gpu,phase_info_gpu->harmonic_normb, pos ,x,y,z,&fx,&fy,&fz);
	
          
    // calculate displacements and proposed positions 
    //! \todo replace for the gaussian rng.
    //const double a = 2*sqrt(1/(3*R));
    //! \todo remove the comment about values in the next line
    // A and R are as in the reference dA =0.002656, dR=0.072887 for coord_huge

    // generate a normal distributed random vector 
	float rx, ry, rz;
	float4 pos_r;

	pos_r=curand_normal4(state);
	
	//soma_normal_vector(&polymers->poly_state_phi,  &rx, &ry, &rz);
	rx=pos_r.x;
	ry=pos_r.y;
	rz=pos_r.z;

    // combine the random offset with the forces, to obtain Brownian motion /
	*dx =A*fx + rx*R;
	*dy =A*fy + ry*R;
	*dz =A*fz + rz*R;

    // calculate proposed position 
	x+=*dx;
	y+=*dy;
	z+=*dz;
	
    // calculate forces in the proposed position 
	nfx=0.0; nfy=0.0;nfz=0.0;
	
	add_bond_forces_arr(poly_type,mono_id,poly_index,poly_arch, phase_info_gpu,phase_info_gpu->harmonic_normb, pos,x,y,z,&nfx,&nfy,&nfz);
	     

    // calculate additional terms for scm energy change 
	*smc_deltaE=0.0;
	*smc_deltaE+=0.5*((nfx+fx)*(*dx) +
                    (nfy+fy)*(*dy) +
                    (nfz+fz)*(*dz));

	*smc_deltaE+=0.25*A*((nfx*nfx)+(nfy*nfy)+(nfz*nfz) -
                      (fx*fx)-(fy*fy)-(fz*fz));



}
__global__ void mc_polymer_move_arr(Phase_info_gpu *phase_info_gpu,Poly_arch *poly_arch,float *pos,curandStatePhilox4_32_10_t *state,double *omega_field_unified){

   
	int poly_id=(blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z)*blockDim.x+threadIdx.x;
	
	
	
	// Rebuild bond information for this chain from bonds, or stay with linear right now?
	
	int poly_index= polymer_number_to_pos(poly_id,0,phase_info_gpu,poly_arch); 
	
	dim3 poly=polymer_number_to_index(poly_id,phase_info_gpu,poly_arch);
	
	int poly_type=poly.x;
        
	float dx=0, dy=0, dz=0;
	double delta_energy=0;

	
	

	//!\todo find a more sophisticated method to select a bead.
	// pick a random bead.

	int length=poly_arch[poly_type].poly_length;
	
	
	

	for(int i=0;i<length;i++){//length
		
		int ibead = curand_uniform_double(&state[poly_id])*length;// ;//% mypoly->poly_arch->poly_length;
		
		if(ibead==length) ibead=length-1;
	
		const unsigned int iwtype =poly_arch[poly_type].Monotype[ibead];
		
		
		double smc_deltaE;
		//if(poly_id==39999) printf("poly_id %d poly_index %d ibead %d\n",poly_id,poly_index,ibead);
        	trial_move_smc_arr(&state[poly_id],pos,poly_id,ibead,poly_type,poly_index,phase_info_gpu,poly_arch,&dx, &dy,& dz,&smc_deltaE);// force biased move
		
				
 		float newx = pos[(poly_index+ibead)*3]+dx;
		float newy = pos[(poly_index+ibead)*3+1]+dy;
 		float newz = pos[(poly_index+ibead)*3+2]+dz;
		//if(poly_id==15000&&i==0) printf("bead %g %g %g\n",pos[(poly_index+ibead)*3],pos[(poly_index+ibead)*3+1],pos[(poly_index+ibead)*3+2]);
		//if(poly_id==15000&&) printf("bead %g %g %g\n",newx,newy,newz);
		int move_allowed =1;
			
		if ( move_allowed  ){

			
              		
			delta_energy =calc_delta_energy_arr(poly_id,ibead,poly_type,pos,phase_info_gpu,poly_arch,omega_field_unified, pos[(poly_index+ibead)*3], pos[(poly_index+ibead)*3+1], pos[(poly_index+ibead)*3+2], dx, dy,dz,iwtype);
			
			delta_energy+=smc_deltaE;

			 // MC roll to accept / reject
			if (som_accept(&state[poly_id] ,delta_energy) == 1) {

				pos[(poly_index+ibead)*3]=newx;
				pos[(poly_index+ibead)*3+1]=newy;
				pos[(poly_index+ibead)*3+2]=newz;
					
						
		        			
						
		 	}//!<  end if som accept
         	}//!< end if move allowed
	

	}//!< end for i
	
	// polymers[npoly]=polymer_de;  
				
__syncthreads();	


  
	   
}

__global__ void test_stream(double *pa){

	int poly_id=(blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z)*blockDim.x+threadIdx.x;

	for(int i=0;i<100000;i++)
		pa[poly_id/100]+=i;

}


