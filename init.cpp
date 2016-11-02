#include "init.h"

int init_scmf(Phase *phase,GPU_info *gpu_info,int argc, char **argv){

	//getCmdLineArgumentString(argc, (const char **) argv, "kernal", &typeChoice);
	

	gpu_info->GPU_N=getCmdLineArgumentInt(argc, (const char **) argv, "GPU_N");

	if(gpu_info->GPU_N==0)gpu_info->GPU_N=1;

	char gpu_chose[20];

	for(int i=0;i<gpu_info->GPU_N;i++){

		sprintf(gpu_chose,"gpu%d",i);

		gpu_info->whichGPUs[i]=i;

		gpu_info->whichGPUs[i]=getCmdLineArgumentInt(argc, (const char **) argv, gpu_chose);

	}
	

	phase->Tcheck=0;

	phase->Tcheck=getCmdLineArgumentInt(argc, (const char **) argv, "Check_T");

	if(phase->Tcheck==0) phase->Tcheck=1000;

	phase->start_time=0;

	phase->start_time=getCmdLineArgumentInt(argc, (const char **) argv, "Start_time");



	phase->N_steps=0;

	phase->N_steps=getCmdLineArgumentInt(argc, (const char **) argv, "Steps_N");
	
	if(phase->N_steps==0) phase->N_steps=100;

	phase->read_file=0;

	phase->read_file=getCmdLineArgumentInt(argc, (const char **) argv, "read_file");


	//printf("start_time %d  N_steps %d Tcheck %d \n",phase->start_time,phase->N_steps,phase->Tcheck);
	

	return 0;
}
