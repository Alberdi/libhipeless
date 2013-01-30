#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "libhipeless.h"

inline void checkErr(cl_int errcode, const char* name) {
  if(errcode != CL_SUCCESS) {
    std::cerr << "ERROR: " << name << " (" << errcode << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
}

const char* readKernelFromSource(const char* source) {
    std::ifstream file(source);
    checkErr(file.is_open() ? CL_SUCCESS : -1, "ifstream()");
    std::string sourceString( std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
    return sourceString.c_str();
}

int opencl_operation(void *TransA, void *TransB, cl_float alpha, const float_matrix *A, const float_matrix *B, cl_float beta, float_matrix *C,
                     int flags, const char* kernelfunction) {
  int i;
  cl_uint num_devices;
  cl_int errcode;
  cl_context context;
  cl_device_id *devices;
  cl_command_queue *command_queues;
  cl_mem memA, memB;
  cl_mem *memC;
  cl_program program;
  cl_kernel kernel;

  int dev_rowsA, last_dev_rowsA, rA;

  const char *source;
  size_t size_devices;
  size_t global_work_size[2];
  size_t local_work_size[2];

  //global_work_size[0] = rowsA + (rowsA % BLOCK_SIZE ? BLOCK_SIZE - (rowsA % BLOCK_SIZE) : 0);
  global_work_size[1] = B->size2 + (B->size2 % BLOCK_SIZE ? BLOCK_SIZE - (B->size2 % BLOCK_SIZE) : 0);
  local_work_size[0] = BLOCK_SIZE;
  local_work_size[1] = BLOCK_SIZE;

  cl_uint size_platforms;
  errcode = clGetPlatformIDs(0, NULL, &size_platforms);
  cl_platform_id* platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id)*size_platforms);
  errcode |= clGetPlatformIDs(size_platforms, platforms, NULL);
  checkErr(errcode, "clGetPlatformIDs");
  // TODO Following line is not applicable to all the possible setups
  cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[flags&USE_CPU ? 0 : 1], 0};

  context = clCreateContextFromType(cps, flags&USE_CPU ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU, NULL, NULL, &errcode);
  checkErr(errcode, "clCreateContextFromType");

  errcode = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);
  checkErr(errcode, "clGetContextInfo1");
  errcode = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size_devices);
  checkErr(errcode, "clGetContextInfo2");
  devices = (cl_device_id *) malloc(size_devices);
  errcode = clGetContextInfo(context, CL_CONTEXT_DEVICES, size_devices, devices, NULL);
  checkErr(errcode, "clGetContextInfo3");

  dev_rowsA = A->size1/num_devices;
  last_dev_rowsA = A->size1 - dev_rowsA*(num_devices-1);

  memA = clCreateBuffer(context, CL_MEM_READ_ONLY, last_dev_rowsA*A->size2*sizeof(cl_float), NULL, &errcode);
  checkErr(errcode, "clCreateBufferA");

  memB = clCreateBuffer(context, CL_MEM_READ_ONLY, B->size1*B->size2*sizeof(cl_float), NULL, &errcode);
  checkErr(errcode, "clCreateBufferB");

  source = readKernelFromSource("operations.cl");
  size_t size_source[] = { strlen(source) };
  program = clCreateProgramWithSource(context, 1, &source, size_source, &errcode);
  checkErr(errcode, "clCreateProgramWithSource");

  errcode = clBuildProgram(program, size_devices/sizeof(cl_device_id), devices, NULL, NULL, NULL);
  checkErr(errcode, "clBuildProgram");

  kernel = clCreateKernel(program, kernelfunction, &errcode);
  checkErr(errcode, "clCreateKernel");
   
  command_queues = (cl_command_queue*) malloc(sizeof(cl_command_queue)*size_devices);
  memC = (cl_mem *) malloc(sizeof(cl_mem)*num_devices);
  for(i=0; i < num_devices; i++) {
    rA = i == num_devices-1 ? last_dev_rowsA : dev_rowsA;
    global_work_size[0] = rA + (rA % BLOCK_SIZE ? BLOCK_SIZE - (rA % BLOCK_SIZE) : 0);

    command_queues[i] = clCreateCommandQueue(context, devices[i], CL_QUEUE_PROFILING_ENABLE, &errcode);
    checkErr(errcode, "clCreateCommandQueue");

    errcode = clEnqueueWriteBuffer(command_queues[i], memA, CL_TRUE, 0,
       rA*A->size2*sizeof(cl_float), &A->data[i*dev_rowsA*A->size2], 0, NULL, NULL);
    checkErr(errcode, "clEnqueueWriteBufferA");

    errcode = clEnqueueWriteBuffer(command_queues[i], memB, CL_TRUE, 0, B->size1*B->size2*sizeof(cl_float), B->data, 0, NULL, NULL);
    checkErr(errcode, "clEnqueueWriteBufferB");

    memC[i] = clCreateBuffer(context, beta ? CL_MEM_READ_WRITE : CL_MEM_WRITE_ONLY, rA*C->size2*sizeof(cl_float), NULL, &errcode);
    checkErr(errcode, "clCreateBufferC");

    if(beta) {
      errcode = clEnqueueWriteBuffer(command_queues[i], memC[i], CL_TRUE, 0, rA*C->size2*sizeof(cl_float), &C->data[i*dev_rowsA*C->size2], 0, NULL, NULL);
      checkErr(errcode, "clEnqueueWriteBufferC");
    }

    errcode = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memC[i]);
    checkErr(errcode, "clSetKernelArg0");

    errcode = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memA);
    checkErr(errcode, "clSetKernelArg1");

    errcode = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memB);
    checkErr(errcode, "clSetKernelArg2");

    errcode = clSetKernelArg(kernel, 3, sizeof(cl_uint), &A->size1);
    checkErr(errcode, "clSetKernelArg3");

    errcode = clSetKernelArg(kernel, 4, sizeof(cl_uint), &A->size2);
    checkErr(errcode, "clSetKernelArg4");

    errcode = clSetKernelArg(kernel, 5, sizeof(cl_uint), &B->size2);
    checkErr(errcode, "clSetKernelArg5");

    errcode = clSetKernelArg(kernel, 6, sizeof(cl_float), &alpha);
    checkErr(errcode, "clSetKernelArg6");

    errcode = clSetKernelArg(kernel, 7, sizeof(cl_float), &beta);
    checkErr(errcode, "clSetKernelArg7");

    errcode = clEnqueueNDRangeKernel(command_queues[i], kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    checkErr(errcode, "clEnqueueNDRangeKernel");
  }

  for(i=0; i < num_devices; i++) {
    rA = i == num_devices-1 ? last_dev_rowsA : dev_rowsA;
    clFinish(command_queues[i]);
    errcode = clEnqueueReadBuffer(command_queues[i], memC[i], CL_TRUE, 0,
      rA*B->size2*sizeof(cl_float), &C->data[i*dev_rowsA*C->size2], 0, NULL, NULL);
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

// C = alpha * A * B + beta * C
// Single precission/float
int blas_sgemm(void* TransA, void* TransB, cl_float alpha, float_matrix *A, float_matrix *B, cl_float beta, float_matrix *C, unsigned int flags) {
//  if(A->size1 != B->size2) { printf("Multiplication not defined for those matrices\n"); return -1; }
//  if(A->size1 != C->size1 || B->size2 != C->size2) { printf("Wrong dimensions on the C matrix\n"); return -1; }

  int root_argument, mpi_size, i, offset;
  int *displs, *scounts;
  unsigned int prows, mrows, saved_Asize1;
  MPI_Comm intercomm, parent;

  if(flags & USE_MPI) {
    char* universe_size = getenv("MPI_UNIVERSE_SIZE");
    if(universe_size == NULL) {
      printf("MPI_UNIVERSE_SIZE is not set\n");
      return -1;
    }
    mpi_size = atoi(universe_size);
    MPI_Comm_get_parent(&parent);
    if(parent == MPI_COMM_NULL) {
      char* mpi_helper = (char *) "mpihelper";
      MPI_Comm_spawn(mpi_helper, MPI_ARGV_NULL, mpi_size-1, MPI_INFO_NULL, 0,
                    MPI_COMM_SELF, &intercomm, MPI_ERRCODES_IGNORE);
      root_argument = MPI_ROOT;
      // prows = processor rows (for each one other than the root)
      prows = A->size1/mpi_size;
      saved_Asize1 = A->size1;
      A->size1 = A->size1 - prows*(mpi_size-1);
    }
    else {
      intercomm = parent;
      root_argument = 0;
      // Matrix allocation
      A = (float_matrix *) malloc(sizeof(float_matrix));
      B = (float_matrix *) malloc(sizeof(float_matrix));
      C = (float_matrix *) malloc(sizeof(float_matrix));
    }

  if(flags & USE_MPI) {
    // Broadcast matrices dimensions
    MPI_Bcast(&prows, 1, MPI_UNSIGNED, root_argument, intercomm);
    MPI_Bcast(&A->size1, 1, MPI_UNSIGNED_LONG, root_argument, intercomm);
    MPI_Bcast(&A->size2, 1, MPI_UNSIGNED_LONG, root_argument, intercomm);
    MPI_Bcast(&B->size1, 1, MPI_UNSIGNED_LONG, root_argument, intercomm);
    MPI_Bcast(&B->size2, 1, MPI_UNSIGNED_LONG, root_argument, intercomm);
    MPI_Bcast(&flags, 1, MPI_UNSIGNED, root_argument, intercomm);
    MPI_Bcast(&alpha, 1, MPI_FLOAT, root_argument, intercomm);
    MPI_Bcast(&beta, 1, MPI_FLOAT, root_argument, intercomm);

    if(parent != MPI_COMM_NULL) {
      A->size1 = prows;
      C->size1 = prows;
      C->size2 = B->size2;
      A->data = (cl_float *) malloc(A->size1*A->size2*sizeof(cl_float));
      B->data = (cl_float *) malloc(B->size1*B->size2*sizeof(cl_float));
      C->data = (cl_float *) malloc(C->size1*C->size2*sizeof(cl_float));
    }

    // Send & Recv A, each node needs prows rows of A
    MPI_Scatter(&A->data[A->size1*A->size2], prows*A->size2, MPI_FLOAT, A->data, A->size1*A->size2, MPI_FLOAT, root_argument, intercomm);
    // Send B in full to each node
    MPI_Bcast(B->data, B->size1*B->size2, MPI_FLOAT, root_argument, intercomm);

    if(beta) {
      // We also need to send C, same rows as A
      MPI_Scatter(&C->data[A->size1*C->size2], prows*C->size2, MPI_FLOAT, C->data, C->size1*C->size2, MPI_FLOAT, root_argument, intercomm);
    }
  }

  opencl_operation(TransA, TransB, alpha, A, B, beta, C, flags, "blas_sgemm");

  if(flags & USE_MPI) {
    // Recv & Send C
    MPI_Gather(C->data, C->size1*C->size2, MPI_FLOAT, &C->data[A->size1*C->size2], prows*C->size2, MPI_FLOAT, root_argument, intercomm);
    // A->size1 might have been overwritten on the parent
    if(parent == MPI_COMM_NULL) {
      A->size1 = saved_Asize1;
    }
  }
}

