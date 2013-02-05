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

int opencl_operation(cl_int nota, cl_int notb, cl_int m, cl_int n, cl_int k, cl_float alpha, cl_float *a, cl_int lda,
                     cl_float *b, cl_int ldb, cl_float beta, cl_float *c, cl_int ldc, unsigned int flags, const char* kernelfunction) {
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

  int dev_i, last_dev_i, iter_i;

  const char *source;
  size_t size_devices;
  size_t global_work_size[2];
  size_t local_work_size[2];

  //global_work_size[0] = rowsA + (rowsA % BLOCK_SIZE ? BLOCK_SIZE - (rowsA % BLOCK_SIZE) : 0);
  global_work_size[1] = n + (n % BLOCK_SIZE ? BLOCK_SIZE - (n % BLOCK_SIZE) : 0);
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

  if(nota) {
    dev_i = m/num_devices;
    last_dev_i = m - dev_i*(num_devices-1);
    memA = clCreateBuffer(context, CL_MEM_READ_ONLY, last_dev_i*k*sizeof(cl_float), NULL, &errcode);
  }
  else {
    dev_i = k/num_devices;
    last_dev_i = k - dev_i*(num_devices-1);
    memA = clCreateBuffer(context, CL_MEM_READ_ONLY, last_dev_i*m*sizeof(cl_float), NULL, &errcode);
  }
  checkErr(errcode, "clCreateBufferA");

  memB = clCreateBuffer(context, CL_MEM_READ_ONLY, k*n*sizeof(cl_float), NULL, &errcode);
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
    iter_i = i == num_devices-1 ? last_dev_i : dev_i;
    global_work_size[0] = iter_i + (iter_i % BLOCK_SIZE ? BLOCK_SIZE - (iter_i % BLOCK_SIZE) : 0);

    command_queues[i] = clCreateCommandQueue(context, devices[i], CL_QUEUE_PROFILING_ENABLE, &errcode);
    checkErr(errcode, "clCreateCommandQueue");

    if(nota) {
      errcode = clEnqueueWriteBuffer(command_queues[i], memA, CL_TRUE, 0, iter_i*k*sizeof(cl_float), &a[i*dev_i*k], 0, NULL, NULL);
    }
    else {
      errcode = clEnqueueWriteBuffer(command_queues[i], memA, CL_TRUE, 0, iter_i*m*sizeof(cl_float), &a[i*dev_i*m], 0, NULL, NULL);
    }
    checkErr(errcode, "clEnqueueWriteBufferA");

    errcode = clEnqueueWriteBuffer(command_queues[i], memB, CL_TRUE, 0, k*n*sizeof(cl_float), b, 0, NULL, NULL);
    checkErr(errcode, "clEnqueueWriteBufferB");

    memC[i] = clCreateBuffer(context, beta ? CL_MEM_READ_WRITE : CL_MEM_WRITE_ONLY, iter_i*n*sizeof(cl_float), NULL, &errcode);
    checkErr(errcode, "clCreateBufferC");

    if(beta) {
      errcode = clEnqueueWriteBuffer(command_queues[i], memC[i], CL_TRUE, 0, iter_i*n*sizeof(cl_float), &c[i*dev_i*n], 0, NULL, NULL);
      checkErr(errcode, "clEnqueueWriteBufferC");
    }
  
    checkErr(clSetKernelArg(kernel, 0, sizeof(cl_int), &nota), "clSetKernelArg0");
    checkErr(clSetKernelArg(kernel, 1, sizeof(cl_int), &notb), "clSetKernelArg1");
    checkErr(clSetKernelArg(kernel, 2, sizeof(cl_int), &m), "clSetKernelArg2");
    checkErr(clSetKernelArg(kernel, 3, sizeof(cl_int), &n), "clSetKernelArg3");
    checkErr(clSetKernelArg(kernel, 4, sizeof(cl_int), &k), "clSetKernelArg4");
    checkErr(clSetKernelArg(kernel, 5, sizeof(cl_float), &alpha), "clSetKernelArg5");
    checkErr(clSetKernelArg(kernel, 6, sizeof(cl_mem), &memA), "clSetKernelArg6");
    checkErr(clSetKernelArg(kernel, 7, sizeof(cl_int), &lda), "clSetKernelArg7");
    checkErr(clSetKernelArg(kernel, 8, sizeof(cl_mem), &memB), "clSetKernelArg8");
    checkErr(clSetKernelArg(kernel, 9, sizeof(cl_int), &ldb), "clSetKernelArg9");
    checkErr(clSetKernelArg(kernel, 10, sizeof(cl_float), &beta), "clSetKernelArg10");
    checkErr(clSetKernelArg(kernel, 11, sizeof(cl_mem), &memC[i]), "clSetKernelArg11");
    checkErr(clSetKernelArg(kernel, 12, sizeof(cl_int), &ldc), "clSetKernelArg12");

    errcode = clEnqueueNDRangeKernel(command_queues[i], kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    checkErr(errcode, "clEnqueueNDRangeKernel");
  }

  for(i=0; i < num_devices; i++) {
    iter_i = i == num_devices-1 ? last_dev_i : dev_i;
    clFinish(command_queues[i]);
    errcode = clEnqueueReadBuffer(command_queues[i], memC[i], CL_TRUE, 0, iter_i*n*sizeof(cl_float), &c[i*dev_i*n], 0, NULL, NULL);
    checkErr(errcode, "clEnqueueReadBuffer");
  }

  for(i=0; i < num_devices; i++) {
    clFinish(command_queues[i]);
    clReleaseCommandQueue(command_queues[i]);
  }

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseContext(context);
}

// C = alpha * A * B + beta * C
// Single precission/float
void blas_sgemm(cl_char transa, cl_char transb, cl_int m, cl_int  n,  cl_int  k,
                cl_float alpha, cl_float *a, cl_int lda, cl_float *b, cl_int ldb,
                cl_float beta, cl_float *c, cl_int ldc, unsigned int flags) {

  int root_argument, mpi_size, nota, notb;
  int rowsa, colsa, rowsb, colsb, spawns_rowsa, spawns_colsa, spawns_m;
  MPI_Comm intercomm, parent;
  MPI_Datatype transtype;

  nota = transa == 'N' || transa == 'n';
  notb = transb == 'N' || transb == 'n';
  if(flags & USE_MPI) {
    char* universe_size = getenv("MPI_UNIVERSE_SIZE");
    if(universe_size == NULL) {
      printf("MPI_UNIVERSE_SIZE is not set\n");
      return;
    }
    mpi_size = atoi(universe_size);
    MPI_Comm_get_parent(&parent);
    if(parent == MPI_COMM_NULL) {
      char* mpi_helper = (char *) "mpihelper";
      MPI_Comm_spawn(mpi_helper, MPI_ARGV_NULL, mpi_size-1, MPI_INFO_NULL, 0,
                    MPI_COMM_SELF, &intercomm, MPI_ERRCODES_IGNORE);
      root_argument = MPI_ROOT;
      if(nota) {
        spawns_rowsa = m/mpi_size;
        spawns_colsa = k;
        rowsa = m - spawns_rowsa*(mpi_size-1);
        colsa = k;
      }
      else {
        spawns_rowsa = k;
        spawns_colsa = m/mpi_size;
        rowsa = k;
        colsa = m - spawns_colsa*(mpi_size-1);
        MPI_Type_vector(rowsa, spawns_colsa, colsa, MPI_FLOAT, &transtype);
        MPI_Type_commit(&transtype);
      }
      if(notb) {
        rowsb = k;
        colsb = n;
      }
      else {
        rowsb = n;
        colsb = k;
      }
    }
    else {
      intercomm = parent;
      root_argument = 0;
    }
  }

  if(flags & USE_MPI) {
    // Broadcast matrices dimensions
    MPI_Bcast(&spawns_rowsa, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&spawns_colsa, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&rowsb, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&colsb, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&alpha, 1, MPI_FLOAT, root_argument, intercomm);
    MPI_Bcast(&beta, 1, MPI_FLOAT, root_argument, intercomm);
    MPI_Bcast(&flags, 1, MPI_UNSIGNED, root_argument, intercomm);
    MPI_Bcast(&nota, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&notb, 1, MPI_INTEGER, root_argument, intercomm);

    if(parent != MPI_COMM_NULL) {
      rowsa = spawns_rowsa;
      colsa = spawns_colsa;
      a = (cl_float *) malloc(rowsa*colsa*sizeof(cl_float));
      b = (cl_float *) malloc(rowsb*colsb*sizeof(cl_float));
      c = (cl_float *) malloc(m*n*sizeof(cl_float));
    }

    if(nota) {
      // Send & Recv A, each node needs spawns_m rows of A with k columns
      MPI_Scatter(&a[rowsa*colsa], spawns_rowsa*spawns_colsa, MPI_FLOAT, a, rowsa*colsa, MPI_FLOAT, root_argument, intercomm);
      m = rowsa;
      k = colsa;
      n = colsb;
      spawns_m = spawns_rowsa;
    }
    else {
      // Send & Recv A, each node needs m rows of A with spawns_k columns
      MPI_Scatter(&a[colsa], 1, transtype, a, rowsa*colsa, MPI_FLOAT, root_argument, intercomm);
      m = colsa;
      k = rowsa;
      n = colsb;
      spawns_m = spawns_colsa;
    }
    // Send B in full to each node
    MPI_Bcast(b, rowsb*colsb, MPI_FLOAT, root_argument, intercomm);

    if(beta) {
      // TODO: transpose with A?
      // We also need to send C, same rows as A
      MPI_Scatter(&c[rowsa*colsa], spawns_rowsa*colsb, MPI_FLOAT, c, rowsa*colsb, MPI_FLOAT, root_argument, intercomm);
    }
  }

  opencl_operation(nota, notb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, flags, "blas_sgemm");

  if(flags & USE_MPI) {
    // Recv & Send C
    //MPI_Gather(c, m*n, MPI_FLOAT, &c[m*n], spawns_m*n, MPI_FLOAT, root_argument, intercomm);
  }
}

