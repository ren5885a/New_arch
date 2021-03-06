#ifndef SOMA_STRUCT_H
#define SOMA_STRUCT_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <stdbool.h>
#include<math.h> 
#include <stdint.h>
#include <curand_kernel.h>

#include <vector>

typedef struct{
	
	int current_node;
	int total_nodes;

}MPI_info; 


/*!\file struct.h
  \brief Definition of all structures related to SOMA.
*/
/*! \brief Monomer struct contains spatial position and type.
  \warning The bit pattern of the type \a w is an int in a double variable.*/

/*! \brief Bond type enumerator to indicate the different bond
 *  types. Matches the bondDict in the ConfGen.py.*/
enum Bondtype{

    HARMONIC=0, /*!<\brief Harmonic bond with a single spring const. */
/*!\deprecated stiff is not implemented, at least for now */
    STIFF=1     /*!<\brief Stiff bonds.*/
};

//! Enum to select the hamiltonian for the non-bonded interaction
enum Hamiltonian{
    SCMF0=0, //!< Original SCMF hamiltonian. For details refer doc of update_omega_fields_scmf0().
    SCMF1=1 //!< Alternative SCMF hamiltonian, especially for more than 2 types. For details refer doc of update_omega_fields_scmf1().
};

enum enum_move_type { move_type__NULL = -1, move_type_arg_TRIAL = 0, move_type_arg_SMART };
enum enum_iteration_alg { iteration_alg__NULL = -1, iteration_alg_arg_POLYMER = 0, iteration_alg_arg_SET };

typedef struct{

	int polymer_type_index; //!< index of the poymer type. 
	int poly_length; //!< number of beads in a polymer.
	int *mono_type_length;

	double length_bond;//!< bond length. 	// default set as 1/31 Re
	int *Monotype;// !< 1D array stores the type of monomer.
	int *connection; // !< 2D array stores the connetction info.
	int *neigh_num;
	int **conection_list; //!< list stores the connection info.
	unsigned int reference_Nbeads;


}Poly_arch;

//! \brief Info needed for output routines.
typedef struct{
    unsigned int delta_mc_Re; //!< \#mc_sweeps between the ana of Re
    unsigned int delta_mc_Rg; //!< \#mc_sweeps between the ana of Rg
    unsigned int delta_mc_b_anisotropy; //!< \#mc_sweeps between the ana of b_aniostropy
    unsigned int delta_mc_density_field;//!< \#mc_sweeps between the ana of density fields
    unsigned int delta_mc_acc_ratio;//!< \#mc_sweeps between the ana of acc ratios
    unsigned int delta_mc_MSD;//!< \#mc_sweeps between the ana of MSD
    unsigned int delta_mc_dump;//!< \#mc_sweeps between full dumps of the coords.
    char * filename; //!< filename of the analysis file.
    char * coord_filename; //!< filename of the configuration files.
}Ana_Info;
/*! \brief Polymer information */
typedef struct{
	double * xn;
	double kn; /*!< This is kn global the trace will contain the scaling of  compressibility \f$\kappa_i\f$ */

	unsigned int nx; /*!< \brief x-spatial discretization */
	unsigned int ny; /*!< \brief y-spatial discretization */
	unsigned int nz; /*!< \brief z-spatial discretization */

	long n_cells; /*!< \brief number of cells in the field */
	double Lx; /*!< \brief x-spatial dimensions in units of \f$ Re_0 \f$ */
	double Ly; /*!< \brief y-spatial dimensions in units of \f$ Re_0 \f$ */
	double Lz; /*!< \brief z-spatial dimensions in units of \f$ Re_0 \f$ */
	double iLx; /*!< \brief inverse x-spatial dimensions in units of \f$ Re_0 \f$ */
	double iLy; /*!< \brief inverse y-spatial dimensions in units of \f$ Re_0 \f$ */
 	double iLz; /*!< \brief inverse z-spatial dimensions in units of \f$ Re_0 \f$ */

	double dx;/*!< length of discrete unit cell in x direction;\f$ */
	double dy;/*!< length of discrete unit cell in y direction;\f$ */
	double dz;/*!< length of discrete unit cell in z direction;\f$ */

	unsigned int n_polymers; /*!< \brief \#polymers in the configuration. */
	unsigned int n_polymers_per_node;	/*!< \brief \#number of polymers in one computation node */
	unsigned int n_polymers_per_gpu;	/*!< \brief \#polymers in the one GPU configuration. */
	
	unsigned int polymer_type_number;	/*!< polymer type number */

	unsigned int *n_polymer_type;	/*!<\brief number of polymers of different types */
	unsigned int *n_polymers_type_per_node;/*!< \brief \#polymers type number in the configuration. */
	unsigned int *n_polymers_type_per_gpu;/*!< \brief \#one type polymers in the one GPU configuration. */

	unsigned int *num_bead_polymer_type; /*!< \brief stores the number of beads of a specific type*/
	unsigned int *num_bead_polymer_type_per_node; /*!< \brief stores the number of beads of a specific type*/
	unsigned int *num_bead_polymer_type_per_gpu; /*!< \brief stores the number of beads of a specific type locally (for this mpi-core)*/

	unsigned int *num_bead_type; /*!< \brief stores the number of beads of a specific type*/
	unsigned int *num_bead_type_per_node; /*!< \brief stores the number of beads of a specific type*/
	unsigned int *num_bead_type_per_gpu; /*!< \brief stores the number of beads of a specific type locally (for this mpi-core)*/

	unsigned int n_mono_types;  /*!<\brief number of monomer types */
	unsigned int num_all_beads; //!< Number of all monomer/beads in the global system
	unsigned int num_all_beads_per_node; //!< Number of all monomer/beads on the local MPI core
	unsigned int num_all_beads_per_gpu; //!< Number of all monomer/beads on the gpu core

	unsigned int *polymer_basis_gpu;
	unsigned int *monomer_poly_basis_gpu;

	double *field_scaling_type; /*< int monomer density field to double density field scalar \f$ */
	double inverse_refbeads; /*< inverese of  reference_Nbeads\f$ */
	int reference_Nbeads; 
	double harmonic_normb;

	double *A;
	double *R;

	uint8_t *area51;	

	
	int *fields_unified;
	int max_safe_jump;

	int current_node;
	int total_nodes;

	int MaxThreadDensity;
	int MaxThreadPolymer;

	
   	
}Phase_info_gpu;

/*! \brief All relevant information for a system configuration.

 * \note Arrays with higher dimensions are usually unrolled as linear
 * arrays for storage. Such unrolled arrays have pointers, which are
 * name with a suffix "_unified". For accessing such pointers, you can
 * use helper function, which are provided by mesh.h.

 * \warning Never access "_unified" pointer arrays, without the helper
 * functions in mesh.h. The memory layout of the array might change
 * with future releases.

 */
typedef struct Phase{
	
	unsigned int reference_Nbeads; /*!< \brief number of reference beads for the model polymer */
	double harmonic_normb; //!< Harmonic energy scale (function of spring constant) const. at runtime.
	double harmonic_spring_Cste; //!< distance harmonic.

	
	
	

    /*! \brief \f$\chi N\f$
      2D matrix with monomer type Flory-Huggins interactions, trace
      contains compressibility \f$\kappa_i\f$.
    */

	double ** xn;
	double kn; /*!< This is kn global the trace will contain the scaling of  compressibility \f$\kappa_i\f$ */


	
	unsigned int n_polymers; /*!< \brief \#polymers in the configuration. */
	unsigned int n_polymers_per_node;	/*!< \brief \#number of polymers in one computation node */
	unsigned int n_polymers_per_gpu;	/*!< \brief \#polymers in the one GPU configuration. */
	
	unsigned int polymer_type_number;	/*!< polymer type number */
	unsigned int *n_polymer_type;	/*!<\brief number of polymers of different types */
	unsigned int *n_polymers_type_per_node;/*!< \brief \#polymers type number in the configuration. */
	unsigned int *n_polymers_type_per_gpu;/*!< \brief \#one type polymers in the one GPU configuration. */

	unsigned int *num_bead_polymer_type; /*!< \brief stores the number of beads of a specific type*/
	unsigned int *num_bead_polymer_type_per_node; /*!< \brief stores the number of beads of a specific type*/
	unsigned int *num_bead_polymer_type_per_gpu; /*!< \brief stores the number of beads of a specific type locally (for this mpi-core)*/


	unsigned int n_mono_types;  /*!<\brief number of monomer types */
	unsigned int *num_bead_type; /*!< \brief stores the number of beads of a specific type*/
	unsigned int *num_bead_type_per_node; /*!< \brief stores the number of beads of a specific type*/
	unsigned int *num_bead_type_per_gpu; /*!< \brief stores the number of beads of a specific type locally (for this mpi-core)*/

	unsigned int num_all_beads; //!< Number of all monomer/beads in the global system
	unsigned int num_all_beads_per_node; //!< Number of all monomer/beads on the local MPI core
	unsigned int num_all_beads_per_gpu; //!< Number of all monomer/beads on the gpu core

	
	std::vector<float *> pos;	/*!< \brief \#position of monomer: n_polymers_per_gpu*poly_length */
	
	
	int gridNx,gridNy,gridNz;
	int GPU_N;
	std::vector<curandStatePhilox4_32_10_t *> state;
	int polymerNx,polymerNy,polymerNz;
	
	
    //uint16_t **fields; /*!< \brief n_types fields in 3D, mimics the DENSITY NOT normalized to 1, this has to be done in the omega_field calculation*/
	std::vector<int*>  fields_unified; /*!< \brief one pointer that points to the construct of p->n_types * p->n_cells of fields */
	std::vector<int*>  fields_32; //!< \brief linear 32 bit version of the fields. This is required for GPU-simulation, because no 16-bit atomic operations are available.

/*! \brief array of shape of field, containing the information if
 * this cell is free space or not. == 0  is a free cell, !=0 is a forbidden cell.
 * If the pointer is set to NULL, the entire simulation box, contains no forbidden cells.
 *
 * \warning area51 is a unified pointer, but is not type specific, so
 * specify for access alway type=0.
 * \todo
 * \warning Before any access to area51, check for NULL.
 */
	std::vector<Phase_info_gpu*> phase_info_gpu;
	std::vector<int*> area51;
	
    //double ** omega_field; /*!< \brief calculates the omega fields according to the Hamiltonian*/
	std::vector<double*> omega_field_unified;  /*!< \brief calculates the omega fields according to the Hamiltonian, unified access*/
	
    //double ** external_field; /*!< \brief external fields that act on the polymers, one field per type */
	std::vector<double*> external_field_unified; /*!< \brief one pointer that points to the construct of p->n_types * p->n_cells of external_fields */
	std::vector<double*> umbrella_field_unified; /*!< \brief one pointer that points to the construct of p->n_types * p->n_cells of umbrella_fields */
	std::vector<uint32_t*> average_field_unified; /*!< \brief one pointer that points to the construct of p->n_types * p->n_cells of average_fields */
	std::vector<uint32_t*> temp_average_field_unified; /*!< \brief one pointer that points to the construct of p->n_types * p->n_cells of temprary average_fields */

	
	
	double * A; /*!< \brief stores the diffusion constants for each type */
	double * R; /*!< \brief stores the derived dR for the diffusion constant */

	double *field_scaling_type; /*!< \brief stores the scaling factor according to the density */

	unsigned int time; /*!< \brief MC steps into the simulation */
	
	unsigned int nx; /*!< \brief x-spatial discretization */
	unsigned int ny; /*!< \brief y-spatial discretization */
	unsigned int nz; /*!< \brief z-spatial discretization */
	unsigned int n_cells; /*!< \brief number of cells in the field */
	double Lx; /*!< \brief x-spatial dimensions in units of \f$ Re_0 \f$ */
	double Ly; /*!< \brief y-spatial dimensions in units of \f$ Re_0 \f$ */
	double Lz; /*!< \brief z-spatial dimensions in units of \f$ Re_0 \f$ */
	double iLx; /*!< \brief inverse x-spatial dimensions in units of \f$ Re_0 \f$ */
	double iLy; /*!< \brief inverse y-spatial dimensions in units of \f$ Re_0 \f$ */
 	double iLz; /*!< \brief inverse z-spatial dimensions in units of \f$ Re_0 \f$ */

	double dx;/*!< length of discrete unit cell in x direction;\f$ */
	double dy;/*!< length of discrete unit cell in y direction;\f$ */
	double dz;/*!< length of discrete unit cell in z direction;\f$ */
    // Variables for statistics/ analytics
	unsigned long int n_moves; /*!< \brief total number of moves */
	unsigned long int n_accepts; /*!< \brief accepted moves */

	double msd_old; /*!< \brief store the MSD from n steps ago */

   	

	Ana_Info ana_info; //!< \brief Bundled info about the output routines.

    
    //! Length of the poly_arch array
	unsigned int poly_arch_length;
   
    	std::vector<Poly_arch*> poly_arch;
	
    //! \brief Struct containing the command line arguments.
    //!struct som_args args;

    //! \brief clock value close to the start of SOMA.
    //!
    //! Not exactly at the start, but at the beginnig of init_values().
	time_t start_clock;
    //! \brief Start time of this simulation.
 	unsigned int start_time;

   
    //! Maximum distance a particle can move, without accidentically
    //!passing trough an area51 wall.
    //!
    //! This is equivalent to min( p->Lx/p->n_x , p->Ly/p->n_y ,
    //! p->Lz/p->n_z). Some versions may introduce some safety
    //! parameter 0 < s < 1 and multiply it with the min.
	double max_safe_jump;

    //! Selected hamiltonian for non-bonded interaction.
	enum Hamiltonian hamiltonian;

    //! Array of independet sets. If NULL no sets are stored, the move type is not available.
    //!
    //! Length of the array is the number of polymer types.
   
    //! Number of max members in an independet set. Used for the length of polymer states.
	unsigned int max_set_members;
    //! Max number of sets for all polymer types.
	unsigned int max_n_sets;
    //! Autotuner for the Monte-Carlo kernels.
    //! Autotuner mc_autotuner;

	
	int MaxThreadDensity;
	int MaxThreadPolymer;
	

	double xiN;
	double kT;
	double dtNxi;
	int Tcheck;
	int Twrite;
	
	int N_steps;
	
	int read_file;	

}Phase;


#endif//SOMA_STRUCT_H
