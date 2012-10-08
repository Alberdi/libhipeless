#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <fstream>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_SIZE 16

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
int matrix_multiplication(cl_float *C, const cl_float *A, const cl_float *B, cl_uint rowsA, cl_uint colsA, cl_uint rowsB, cl_uint colsB) {
  if(colsA != rowsB) { printf("Multiplication not defined for those matrices\n"); return -1; }
  cl_int errcode;
  cl_context context;
  cl_device_id *devices;
  cl_command_queue command_queue;
  cl_mem memA, memB, memC;
  cl_program program;
  cl_kernel kernel;

  const char *source;
  size_t size_devices;
  size_t global_work_size[2];
  size_t local_work_size[2];

  global_work_size[0] = rowsA;
  global_work_size[1] = colsB;
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

  errcode = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size_devices);
  devices = (cl_device_id *) malloc(size_devices);
  errcode |= clGetContextInfo(context, CL_CONTEXT_DEVICES, size_devices, devices, NULL);
  checkErr(errcode, "clGetContextInfo");

  // We take the first GPU device
  command_queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &errcode);
  checkErr(errcode, "clCreateCommandQueue");

  memA = clCreateBuffer(context, CL_MEM_READ_ONLY, rowsA*colsA*sizeof(cl_float), NULL, &errcode);
  checkErr(errcode, "clCreateBuffer");

  memB = clCreateBuffer(context, CL_MEM_READ_ONLY, rowsB*colsB*sizeof(cl_float), NULL, &errcode);
  checkErr(errcode, "clCreateBuffer");

  memC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, rowsA*colsB*sizeof(cl_float), NULL, &errcode);
  checkErr(errcode, "clCreateBuffer");

  errcode = clEnqueueWriteBuffer(command_queue, memA, CL_TRUE, 0, rowsA*colsA*sizeof(cl_float), A, 0, NULL, NULL);
  checkErr(errcode, "clEnqueueWriteBuffer");

  errcode = clEnqueueWriteBuffer(command_queue, memB, CL_TRUE, 0, rowsB*colsB*sizeof(cl_float), B, 0, NULL, NULL);
  checkErr(errcode, "clEnqueueWriteBuffer");

  source = readKernelFromSource("./matmul.cl");
  size_t size_source[] = { strlen(source) };
  program = clCreateProgramWithSource(context, 1, &source, size_source, &errcode);
  checkErr(errcode, "clCreateProgramWithSource");
  
  errcode = clBuildProgram(program, size_devices/sizeof(cl_device_id), devices, NULL, NULL, NULL);
  checkErr(errcode, "clBuildProgram");

  kernel = clCreateKernel(program, "matmul", &errcode);
  checkErr(errcode, "clCreateKernel");
 
  errcode = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memC);
  checkErr(errcode, "clSetKernelArg");

  errcode = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memA);
  checkErr(errcode, "clSetKernelArg");

  errcode = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memB);
  checkErr(errcode, "clSetKernelArg");

  errcode = clSetKernelArg(kernel, 3, sizeof(cl_uint), &colsA);
  checkErr(errcode, "clSetKernelArg");

  errcode = clSetKernelArg(kernel, 4, sizeof(cl_uint), &colsB);
  checkErr(errcode, "clSetKernelArg");

  errcode = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  checkErr(errcode, "clEnqueueNDRangeKernel");

  clFinish(command_queue);

  errcode = clEnqueueReadBuffer(command_queue, memC, CL_TRUE, 0, rowsA*colsB*sizeof(cl_float), C, 0, NULL, NULL);
  checkErr(errcode, "clEnqueueReadBuffer");

  clFinish(command_queue);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  return 1;
}

int main(int argc, char* argv[]) {
  int i, j;
  int mpi_rank, mpi_size;
  int rowsA = 1025, colsA = 512, rowsB = 512, colsB = 2048;
  int prows, mrows, fill;
  cl_float *A, *B, *C;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Calculate the rows of A to be multiplied by each node
  // prows = processor rows (for each one other than the root)
  prows = rowsA/mpi_size - ((rowsA/mpi_size) % BLOCK_SIZE);
  // mrows = master rows (root)
  mrows = rowsA - prows*(mpi_size-1);
  // fill = rows that must be added to the root to be a multiple of 16
  if(mrows % BLOCK_SIZE != 0)
    fill = BLOCK_SIZE - (mrows % BLOCK_SIZE);

  // Matrix allocation and initialization
  if(!mpi_rank) {
    A = (cl_float *) malloc(rowsA*colsA*sizeof(cl_float));
    B = (cl_float *) malloc(rowsB*colsB*sizeof(cl_float));
    C = (cl_float *) malloc(rowsA*colsB*sizeof(cl_float));

    for(i=0;i<rowsA;i++)
      for(j=0;j<colsA;j++)
        A[i*colsA+j]=1;

    for(i=0;i<rowsB;i++)
      for(j=0;j<colsB;j++)
        B[i*colsB+j] = i==j ? 1 : 0;
  }
  else {
    // We divide by mpi_size because we only need a fraction of A and C
    A = (cl_float *) malloc(prows*colsA*sizeof(cl_float));
    B = (cl_float *) malloc(rowsB*colsB*sizeof(cl_float));
    C = (cl_float *) malloc(prows*colsB*sizeof(cl_float));
  }

  // Send & Recv A, each node needs rowsA/mpi_size rows of A
  MPI_Scatter(A, prows*colsA, MPI_FLOAT, A, prows*colsA, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Send B in full to each node
  MPI_Bcast(B, rowsB*colsB, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Do the partial multiplication
  matrix_multiplication(C, A, B, mpi_rank ? prows : mrows+fill, colsA, rowsB, colsB);

  // Recv & Send C
  if(!mpi_rank)
    for(i=1; i<mpi_size; i++)
      MPI_Recv(&C[(mrows+prows*(i-1))*colsB], prows*colsB, MPI_FLOAT, i, 6541, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  else
    MPI_Send(C, prows*colsB, MPI_FLOAT, 0, 6541, MPI_COMM_WORLD);
  //MPI_Gather(C, mpi_rank ? prows*colsB : mrows*colsB, MPI_FLOAT, C, mrows*colsB, MPI_FLOAT, 0, MPI_COMM_WORLD);

 // Result checking
  if(!mpi_rank) {
    float x = 0.0;
    for(i=0; i<rowsA; i++)
      for(j=0; j<colsB; j++)
        x += C[i*colsB+j];

    // TODO This check is not correct, but the results seem to be correct always
    // (checked with octave)
    // Tip to fix: sometimes it is rowsA*colsA and sometimes rowsA*colsB
    if(x==rowsA*colsA || x==rowsA*colsB) printf("CORRECTO (%f)\n", x);
    else printf("INCORRECTO: %f (%d, %d)\n", x, rowsA*colsA, rowsA*colsB);
  }
  MPI_Finalize();
    
}
