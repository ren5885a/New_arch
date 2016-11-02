#include <stdio.h>
#include "struct.h"


#include"mpi_info.h"
#include  "test.h"
int main(int argc, char *argv[]){

	Phase phase;
	MPI_info mpi_info;

	mpi_initialize(argc,argv,&mpi_info);
	
	test_program( &mpi_info,&phase,argc, argv);
	

	MPI_Finalize(); 
	return 0;
}
