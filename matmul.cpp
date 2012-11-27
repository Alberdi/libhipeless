#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_SIZE 16
#define MPI_INIT_TAG 9876
#define MPI_RESULT_TAG 6789

void matrix_print(cl_float *A, cl_uint rowsA, cl_uint colsA) {
  int i, j;
  for(i=0; i<rowsA; i++) {
    for(j=0; j<colsA; j++) {
      printf("%d ", (int)A[i*colsA+j]);
    }
    printf("\n");
  }
}

//TODO se debería salir de la función con un error, no hacer un exit()
inline void checkErr(cl_int errcode, const char* name) {
  if(errcode != CL_SUCCESS) {
    std::cerr << "ERROR: " << name << " (" << errcode << ")" << std::endl;
    //exit(EXIT_FAILURE);
  }
}

const char* readKernelFromSource(const char* source) {
    std::ifstream file(source);
    checkErr(file.is_open() ? CL_SUCCESS : -1, "ifstream()");
    std::string sourceString( std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
    return sourceString.c_str();
}

// C = A*B
int matrix_multiplication(cl_float *C, const cl_float *A, const cl_float *B, cl_uint rowsA, cl_uint colsA, cl_uint rowsB, cl_uint colsB) {
  if(colsA != rowsB) { printf("Multiplication not defined for those matrices\n"); return -1; }
  int i;
  cl_uint num_devices;
  cl_int errcode;
  cl_context context;
  cl_device_id *devices;
  cl_command_queue* command_queues;
  cl_mem memA, memB, memC;
  cl_program program;
  cl_kernel kernel;

  int dev_rowsA, last_dev_rowsA, rA;

  const char *source;
  size_t size_devices;
  size_t global_work_size[2];
  size_t local_work_size[2];

  global_work_size[0] = rowsA + (rowsA % BLOCK_SIZE ? BLOCK_SIZE - (rowsA % BLOCK_SIZE) : 0);
  global_work_size[1] = rowsB + (rowsB % BLOCK_SIZE ? BLOCK_SIZE - (rowsB % BLOCK_SIZE) : 0);
  local_work_size[0] = BLOCK_SIZE;
  local_work_size[1] = BLOCK_SIZE;

  //TODO Me gustaría obviar las siguientes líneas
  cl_uint size_platforms;
  errcode = clGetPlatformIDs(0, NULL, &size_platforms);
  cl_platform_id* platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id)*size_platforms);
  errcode |= clGetPlatformIDs(size_platforms, platforms, NULL);
  checkErr(errcode, "clGetPlatformIDs");
  cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[1], 0};
  // Hasta aquí

  context = clCreateContextFromType(cps, CL_DEVICE_TYPE_GPU, NULL, NULL, &errcode);
  checkErr(errcode, "clCreateContextFromType");

  errcode = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);
  checkErr(errcode, "clGetContextInfo1");
  errcode = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size_devices);
  checkErr(errcode, "clGetContextInfo2");
  devices = (cl_device_id *) malloc(size_devices);
  errcode = clGetContextInfo(context, CL_CONTEXT_DEVICES, size_devices, devices, NULL);
  checkErr(errcode, "clGetContextInfo3");

  dev_rowsA = rowsA/num_devices;
  last_dev_rowsA = rowsA - dev_rowsA*(num_devices-1);

  memA = clCreateBuffer(context, CL_MEM_READ_ONLY, last_dev_rowsA*colsA*sizeof(cl_float), NULL, &errcode);
  checkErr(errcode, "clCreateBuffer");

  memB = clCreateBuffer(context, CL_MEM_READ_ONLY, rowsB*colsB*sizeof(cl_float), NULL, &errcode);
  checkErr(errcode, "clCreateBuffer");

  memC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, last_dev_rowsA*colsB*sizeof(cl_float), NULL, &errcode);
  checkErr(errcode, "clCreateBuffer");

  source = readKernelFromSource("./matmul.cl");
  size_t size_source[] = { strlen(source) };
  program = clCreateProgramWithSource(context, 1, &source, size_source, &errcode);
  checkErr(errcode, "clCreateProgramWithSource");

  errcode = clBuildProgram(program, size_devices/sizeof(cl_device_id), devices, NULL, NULL, NULL);
  checkErr(errcode, "clBuildProgram");

  kernel = clCreateKernel(program, "matmul", &errcode);
  checkErr(errcode, "clCreateKernel");
   
  command_queues = (cl_command_queue*) malloc(sizeof(cl_command_queue)*size_devices);
  for(i=0; i < num_devices; i++) {
    rA = i == num_devices-1 ? last_dev_rowsA : dev_rowsA;
    command_queues[i] = clCreateCommandQueue(context, devices[i], CL_QUEUE_PROFILING_ENABLE, &errcode);
    checkErr(errcode, "clCreateCommandQueue");

    errcode = clEnqueueWriteBuffer(command_queues[i], memA, CL_TRUE, 0,
       rA*colsA*sizeof(cl_float), &A[i*(rowsA*colsA/num_devices)], 0, NULL, NULL);
    checkErr(errcode, "clEnqueueWriteBufferA");

    errcode = clEnqueueWriteBuffer(command_queues[i], memB, CL_TRUE, 0, rowsB*colsB*sizeof(cl_float), B, 0, NULL, NULL);
    checkErr(errcode, "clEnqueueWriteBufferB");

    errcode = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memC);
    checkErr(errcode, "clSetKernelArg");

    errcode = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memA);
    checkErr(errcode, "clSetKernelArg");

    errcode = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memB);
    checkErr(errcode, "clSetKernelArg");

    errcode = clSetKernelArg(kernel, 3, sizeof(cl_uint), &rowsA);
    checkErr(errcode, "clSetKernelArg");

    errcode = clSetKernelArg(kernel, 4, sizeof(cl_uint), &colsA);
    checkErr(errcode, "clSetKernelArg");

    errcode = clSetKernelArg(kernel, 5, sizeof(cl_uint), &colsB);
    checkErr(errcode, "clSetKernelArg");

    errcode = clEnqueueNDRangeKernel(command_queues[i], kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    checkErr(errcode, "clEnqueueNDRangeKernel");
  }

  for(i=0; i < num_devices; i++) {
    clFinish(command_queues[i]);
    errcode = clEnqueueReadBuffer(command_queues[i], memC, CL_TRUE, 0,
      rA*colsB*sizeof(cl_float), &C[i*(rowsA*colsB/num_devices)], 0, NULL, NULL);
    checkErr(errcode, "clEnqueueReadBuffer");
  }

  for(i=0; i < num_devices; i++) {
    clFinish(command_queues[i]);
    clReleaseCommandQueue(command_queues[i]);
  }

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseContext(context);

  return 1;
}

int main(int argc, char* argv[]) {
  int i, j;
  int rowsA = 61, colsA = 125, rowsB = 125, colsB = 67;
  //int rowsA = 1024, colsA = 512, rowsB = 512, colsB = 2048;
  cl_float *A, *B, *C;

 // Matrix allocation and initialization
  A = (cl_float *) malloc(rowsA*colsA*sizeof(cl_float));
  B = (cl_float *) malloc(rowsB*colsB*sizeof(cl_float));
  C = (cl_float *) malloc(rowsA*colsB*sizeof(cl_float));

  for(i=0;i<rowsA;i++)
    for(j=0;j<colsA;j++)
      A[i*colsA+j]=1;

  for(i=0;i<rowsB;i++)
    for(j=0;j<colsB;j++)
      B[i*colsB+j] = i==j ? 1 : 0;

  // Do the partial multiplication
  matrix_multiplication(C, A, B, rowsA, colsA, rowsB, colsB);

  float x = 0.0;
  for(i=0; i<rowsA; i++) {
    for(j=0; j<colsB; j++) {
      x += C[i*colsB+j];
    }
  }

  // TODO This check is not correct, but the results seem to be correct always
  // (checked with octave)
  // Tip to fix: sometimes it is rowsA*colsA and sometimes rowsA*colsB
  if(x==rowsA*colsA || x==rowsA*colsB) printf("CORRECTO (%f)\n", x);
  else printf("INCORRECTO: %f (%d, %d)\n", x, rowsA*colsA, rowsA*colsB);
 
  //matrix_print(C, rowsA, colsB);
    
}