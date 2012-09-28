#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#define NN 1024
#include <string.h>
#include <fstream>
#include <iostream>
#include <time.h>


inline void
checkErr(cl_int err, const char * name)
{
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " << name
		<< " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}


void showAvailablePlatforms(cl_uint num_platforms, cl_platform_id *platforms) {
	int i;
	size_t param_value_size;
	char *param_value;
	cl_int err;
	
	
	for(i=0;i<num_platforms;i++) {
		
		std::cout << "================" << std::endl; 
		std::cout << "Plataforma " << i << std::endl;
		
		err=clGetPlatformInfo(platforms[i],CL_PLATFORM_NAME,0, NULL,&param_value_size);
		checkErr(err,"clGetPlatformInfo");
		param_value=(char *) malloc(param_value_size);
		err=clGetPlatformInfo(platforms[i],CL_PLATFORM_NAME,param_value_size, param_value,NULL);
		checkErr(err,"clGetPlatformInfo");
		std::cout << "Nombre : " << param_value << std::endl;
		
		err=clGetPlatformInfo(platforms[i],CL_PLATFORM_VENDOR, 0,NULL,&param_value_size);
		checkErr(err,"clGetPlatformInfo");
		param_value=(char *) malloc(param_value_size);
		err=clGetPlatformInfo(platforms[i],CL_PLATFORM_VENDOR,param_value_size, param_value,NULL);
		checkErr(err,"clGetPlatformInfo");
		std::cout << "Fabricante : " << param_value << std::endl;
		
		err=clGetPlatformInfo(platforms[i],CL_PLATFORM_VERSION,0, NULL,&param_value_size);
		checkErr(err,"clGetPlatformInfo");
		param_value=(char *) malloc(param_value_size);
		err=clGetPlatformInfo(platforms[i],CL_PLATFORM_VERSION,param_value_size, param_value,NULL);
		checkErr(err,"clGetPlatformInfo");
		std::cout << "Versión : " << param_value << std::endl;
		
		std::cout << "================" << std::endl; 
		
	}
	
}

void showAvailableDevices(cl_uint num_devices, cl_device_id *devices) {
	int i;
	size_t param_value_size;
	char *param_value;
	cl_int err;
	
	
	for(i=0;i<num_devices;i++) {
		
		std::cout << "================" << std::endl; 
		std::cout << "Dispositivo " << i << std::endl;
		
		err=clGetDeviceInfo(devices[i],CL_DEVICE_NAME, 0,NULL,&param_value_size);
		checkErr(err,"clGetDeviceInfo");
		param_value=(char *) malloc(param_value_size);
		err=clGetDeviceInfo(devices[i],CL_DEVICE_NAME,param_value_size, param_value,NULL);
		checkErr(err,"clGetDeviceInfo");
		std::cout << "Nombre : " << param_value << std::endl;
		
		std::cout << "================" << std::endl; 
	}
	
}


int main(int argc, char *argv[]) {
	cl_int err;
	cl_context context;
	cl_device_id *devices;
	size_t size_devices;
	cl_int num_devices;
	cl_platform_id *platforms;
	cl_uint num_platforms;
	cl_context_properties properties;
	cl_mem memA,memB,memC;
	size_t N=NN;
	size_t *global_work_size;
	size_t *local_work_size;
	unsigned int i,j,k,selected_platform,selected_device;
	cl_float x;
	
	global_work_size=(size_t *) malloc(2*sizeof(size_t));
	local_work_size=(size_t *) malloc(2*sizeof(size_t));
	
	global_work_size[0]=N;
	local_work_size[0]=16;
	global_work_size[1]=N;
	local_work_size[1]=16;	
	
	cl_float *A=(cl_float *) malloc(N*N*sizeof(cl_float));
	cl_float *B=(cl_float *) malloc(N*N*sizeof(cl_float));
	cl_float *C=(cl_float *) malloc(N*N*sizeof(cl_float));
	
	for(i=0;i<N;i++)
		for(j=0;j<N;j++){
			B[i*N+j]=1;
			if (i==j) C[i*N+j]=1; else C[i*N+j]=0;
		}
		
		
	
	err= clGetPlatformIDs(0, NULL, &num_platforms);
	
	checkErr(err,"clGetPlatformIDs");
	
	std::cout << "Detectada(s) " << num_platforms << " plataforma(s)" << std::endl;
	
	platforms=(cl_platform_id *) malloc(sizeof(cl_platform_id)*num_platforms);
	
	err= clGetPlatformIDs(num_platforms, platforms, NULL);
	
	checkErr(err,"clGetPlatformIDs");
	
	showAvailablePlatforms(num_platforms,platforms);
	
	selected_platform=1;
	
	std::cout << "Seleccionamos la plataforma " << selected_platform << std::endl;
	
	std::cout << "Creamos un contexto para los dispotivos de tipo GPU presentes en la plataforma " << std::endl;
	
	// Creamos un contexto con GPUs
	cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[selected_platform], 0};	
	context=clCreateContextFromType(cps, CL_DEVICE_TYPE_ALL,NULL,NULL,&err);
	checkErr(err,"clCreateContextFromType");
	
	
	err = clGetContextInfo(context,CL_CONTEXT_DEVICES,0,NULL,&size_devices);
	checkErr(err,"clGetContextInfo");
	
	num_devices=size_devices/sizeof(cl_device_id);
	
	std::cout << "Detectado(s) " << num_devices << " dispositivo(s) de tipo GPU" << " en la plataforma " << selected_platform << std::endl;
	
	devices= (cl_device_id*) malloc(size_devices);
	
	err = clGetContextInfo(context,CL_CONTEXT_DEVICES,size_devices,devices,NULL);
	checkErr(err,"clGetContextInfo");
	
	showAvailableDevices(num_devices,devices);
	
	selected_device=0;
	
	cl_command_queue command_queue=clCreateCommandQueue(context, devices[selected_device], CL_QUEUE_PROFILING_ENABLE,NULL);
	checkErr(err,"clCreateCommandQueue");
	
	std::cout << "Creamos una cola de comandos para el dispositivo " << selected_device << std::endl;
	
	memA=clCreateBuffer(context, CL_MEM_WRITE_ONLY ,N*N*sizeof(cl_float),NULL,NULL);
	checkErr(err,"clCreateBuffer");
	
	memB=clCreateBuffer(context, CL_MEM_READ_ONLY,N*N*sizeof(cl_float),NULL,NULL);
	checkErr(err,"clCreateBuffer");
	
	memC=clCreateBuffer(context, CL_MEM_READ_ONLY,N*N*sizeof(cl_float),NULL,NULL);
	checkErr(err,"clCreateBuffer");
	
	err=clEnqueueWriteBuffer(command_queue,memB,CL_TRUE, 0,N*N*sizeof(cl_float),B,0,NULL,NULL);
	checkErr(err,"clEnqueueWriteBuffer");
	
	err=clEnqueueWriteBuffer(command_queue,memC,CL_TRUE, 0,N*N*sizeof(cl_float),C,0,NULL,NULL);
	checkErr(err,"clEnqueueWriteBuffer");
	
	//Leemos el código del kernel a ejecutar de un fichero
	std::ifstream file("./matmul.cl");
	checkErr(file.is_open() ? CL_SUCCESS : -1, "ifstream() no puede acceder al fichero");
	std::string sourceString( std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
	
	//Convertimos el string leído a char *
     	const char *source = sourceString.c_str();
     	size_t sourceSize[]={ strlen (source) };
	
	cl_program program = clCreateProgramWithSource(context,1,&source,sourceSize,&err);
	checkErr(err,"clCreateProgramWithSource");
	
	
	
	// Creo un ejecutable para todos los dispositivos disponibles
	err = clBuildProgram(program, num_devices, devices, NULL, NULL, NULL);
	
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: Compilación del Kernel" << std::endl;
		char* build_log;
		size_t log_size;
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		build_log = new char[log_size+1];
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
		build_log[log_size] = '\0';
		std::cout << build_log << std::endl;
		delete[] build_log;
		exit(EXIT_FAILURE);
	}
	
	// Creo un kernel usando el programa que acabo de construir
	cl_kernel kernel=clCreateKernel(program,"matmul",&err);
	checkErr(err,"clCreateKernel");
	
	//Al kernel le paso como argumento los 2 vectores de entrada, el vector resultado y el tamaño del vector
	err=clSetKernelArg(kernel,0,sizeof(cl_mem),&memA);
	checkErr(err,"clSetKernelArg");
	
	err=clSetKernelArg(kernel,1,sizeof(cl_mem),&memB);
	checkErr(err,"clSetKernelArg");
	
	err=clSetKernelArg(kernel,2,sizeof(cl_mem),&memC);
	checkErr(err,"clSetKernelArg");
	
	err=clSetKernelArg(kernel,3,sizeof(cl_uint),&N);
	checkErr(err,"clSetKernelArg");
	
	cl_event event;
	//Pongo la ejecución del kernel en la cola especificando el tamaño del grupo de trabajo y el tamaño de cada grupo
	err=clEnqueueNDRangeKernel(command_queue,kernel,2,NULL,global_work_size,local_work_size,0,NULL,&event);
	checkErr(err,"clEnqueueNDRangeKernel");

	clFinish(command_queue);

	//Leo el resultado
	err=clEnqueueReadBuffer(command_queue,memA,CL_TRUE, 0,N*N*sizeof(cl_float),A,0,NULL,NULL);
	checkErr(err,"clEnqueueReadBuffer");
	
	
	//Cerramos la cola de comandos y liberamos la cola de comandos, el kernel, el programa y el contexto
	clFinish(command_queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	//Compruebo el resultado
	x=0.0;
	
	for(i=0;i<N;i++)
		for(j=0;j<N;j++)
			x+=A[i*N+j];
	
	if (x==(N*N)) printf("Resultado CORRECTO\n");
	else printf("Resultado INCORRECTO %f\n",x);
	
	return 1;
}



	


	

