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
#include <mpi.h>

#define BLOCK_SIZE 16

#define USE_CPU 0x01
#define USE_GPU 0x02
#define USE_MPI 0x04
#define NON_MPI_ROOT 0x08

void matrix_print(cl_float *A, cl_uint rowsA, cl_uint colsA) {
  int i, j;
  for(i=0; i<rowsA; i++) {
    for(j=0; j<colsA; j++) {
      printf("%d ", (int)A[i*colsA+j]);
    }
    printf("\n");
  }
}

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

// C = A*B
int matrix_multiplication_cl(cl_float *C, const cl_float *A, const cl_float *B, cl_uint rowsA, cl_uint colsA, cl_uint rowsB, cl_uint colsB, unsigned int flags) {
  if(colsA != rowsB) { printf("Multiplication not defined for those matrices\n"); return -1; }
  int i;
  cl_uint num_devices;
  cl_int errcode;
  cl_context context;
  cl_device_id *devices;
  cl_command_queue* command_queues;
  cl_mem memA, memB;
  cl_mem* memC;
  cl_program program;
  cl_kernel kernel;

  int dev_rowsA, last_dev_rowsA, rA;

  const char *source;
  size_t size_devices;
  size_t global_work_size[2];
  size_t local_work_size[2];

  //global_work_size[0] = rowsA + (rowsA % BLOCK_SIZE ? BLOCK_SIZE - (rowsA % BLOCK_SIZE) : 0);
  global_work_size[1] = colsB + (colsB % BLOCK_SIZE ? BLOCK_SIZE - (colsB % BLOCK_SIZE) : 0);
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

  dev_rowsA = rowsA/num_devices;
  last_dev_rowsA = rowsA - dev_rowsA*(num_devices-1);

  memA = clCreateBuffer(context, CL_MEM_READ_ONLY, last_dev_rowsA*colsA*sizeof(cl_float), NULL, &errcode);
  checkErr(errcode, "clCreateBufferA");

  memB = clCreateBuffer(context, CL_MEM_READ_ONLY, rowsB*colsB*sizeof(cl_float), NULL, &errcode);
  checkErr(errcode, "clCreateBufferB");

  memC = (cl_mem *) malloc(sizeof(cl_mem)*num_devices);
  for(i=0; i < num_devices; i++) {
    rA = i == num_devices-1 ? last_dev_rowsA : dev_rowsA;
    memC[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, rA*colsB*sizeof(cl_float), NULL, &errcode);
    checkErr(errcode, "clCreateBufferC");
  }

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
    global_work_size[0] = rA + (rA % BLOCK_SIZE ? BLOCK_SIZE - (rA % BLOCK_SIZE) : 0);

    command_queues[i] = clCreateCommandQueue(context, devices[i], CL_QUEUE_PROFILING_ENABLE, &errcode);
    checkErr(errcode, "clCreateCommandQueue");

    errcode = clEnqueueWriteBuffer(command_queues[i], memA, CL_TRUE, 0,
       rA*colsA*sizeof(cl_float), &A[i*dev_rowsA*colsA], 0, NULL, NULL);
    checkErr(errcode, "clEnqueueWriteBufferA");

    errcode = clEnqueueWriteBuffer(command_queues[i], memB, CL_TRUE, 0, rowsB*colsB*sizeof(cl_float), B, 0, NULL, NULL);
    checkErr(errcode, "clEnqueueWriteBufferB");

    errcode = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memC[i]);
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
    rA = i == num_devices-1 ? last_dev_rowsA : dev_rowsA;
    clFinish(command_queues[i]);
    errcode = clEnqueueReadBuffer(command_queues[i], memC[i], CL_TRUE, 0,
      rA*colsB*sizeof(cl_float), &C[i*dev_rowsA*colsB], 0, NULL, NULL);
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

// C = A*B
int matrix_multiplication(cl_float *C, cl_float *A, cl_float *B, cl_uint rowsA, cl_uint colsA, cl_uint rowsB, cl_uint colsB,
                          unsigned int flags) {
  int root_argument, mpi_size;
  int prows, mrows;
  MPI_Comm intercomm, parent;

  if(flags & USE_MPI) {
    char* universe_size = getenv("MPI_UNIVERSE_SIZE");
    if(universe_size == NULL) {
      printf("MPI_UNIVERSE_SIZE is not set\n");
      return -1;
    }
    mpi_size = atoi(universe_size);
    MPI_Init(0, NULL);
    MPI_Comm_get_parent(&parent);
    if(parent == MPI_COMM_NULL) {
      char* mpi_helper = (char *) "mpihelper";
      MPI_Comm_spawn(mpi_helper, MPI_ARGV_NULL, mpi_size-1, MPI_INFO_NULL, 0,
                    MPI_COMM_SELF, &intercomm, MPI_ERRCODES_IGNORE);
      root_argument = MPI_ROOT;
      // prows = processor rows (for each one other than the root)
      prows = rowsA/mpi_size;
      // mrows = master rows (root)
      mrows = rowsA - prows*(mpi_size-1);
    }
    else {
      intercomm = parent;
      root_argument = 0;
    }
  } else {
    mrows = rowsA;
    root_argument = 1;
  }

  if(flags & USE_MPI) {
    // Broadcast matrices dimensions
    MPI_Bcast(&prows, 1, MPI_UNSIGNED, root_argument, intercomm);
    MPI_Bcast(&colsA, 1, MPI_UNSIGNED, root_argument, intercomm);
    MPI_Bcast(&rowsB, 1, MPI_UNSIGNED, root_argument, intercomm);
    MPI_Bcast(&colsB, 1, MPI_UNSIGNED, root_argument, intercomm);
    MPI_Bcast(&flags, 1, MPI_UNSIGNED, root_argument, intercomm);

    if (parent != MPI_COMM_NULL) {
      // Matrix allocation
      A = (cl_float *) malloc(prows*colsA*sizeof(cl_float));
      B = (cl_float *) malloc(rowsB*colsB*sizeof(cl_float));
      C = (cl_float *) malloc(prows*colsB*sizeof(cl_float));
    }

    // Send & Recv A, each node needs prows rows of A
    MPI_Scatter(A, prows*colsA, MPI_FLOAT, A, prows*colsA, MPI_FLOAT, root_argument, intercomm);
    // Send B in full to each node
    MPI_Bcast(B, rowsB*colsB, MPI_FLOAT, root_argument, intercomm);
  }

  matrix_multiplication_cl(C, A, B, root_argument ? mrows : prows, colsA, rowsB, colsB, flags);

  if(flags & USE_MPI) {
    // Recv & Send C
    MPI_Gather(C, prows*colsB, MPI_FLOAT, C, prows*colsB, MPI_FLOAT, root_argument, intercomm);
    MPI_Finalize();
  }
}

// C = alpha*B
int scalar_matrix_multiplication(cl_float *C, cl_float alpha, cl_float *B, cl_uint rowsB, cl_uint colsB, unsigned int flags) {
  matrix_multiplication(C, &alpha, B, 1, 1, rowsB, colsB, flags);
}

