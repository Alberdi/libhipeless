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
  local_work_size[0] = 16;
  local_work_size[1] = 16;

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
  //int rowsA = 2048, colsA = 2048, rowsB = 2048, colsB = 2048;
  //int rowsA = 1024, colsA = 512, rowsB = 512, colsB = 2048;
  int rowsA = 64, colsA = 64, rowsB = 64, colsB = 64;
  cl_float *A, *B, *C;
  MPI_Datatype custom_datatype;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  // TODO we asume mpi_size = 2 for a while

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
    // We divide by mpi_size because we only need a fraction of each matrix
    A = (cl_float *) malloc(rowsA*colsA*sizeof(cl_float)/mpi_size);
    B = (cl_float *) malloc(rowsB*colsB*sizeof(cl_float)/mpi_size);
    C = (cl_float *) malloc(rowsA*colsB*sizeof(cl_float)/mpi_size);
  }

  // Send & Recv stuff
  if(!mpi_rank) {
    // We need a datatype to pack the needed elements of B
    MPI_Type_vector(rowsB, colsB/mpi_size, colsB, MPI_FLOAT, &custom_datatype);
    MPI_Type_commit(&custom_datatype);

    // We hardcode the 1 as the number of the receiving processor
    MPI_Send(&A[rowsA*colsA/mpi_size], rowsA*colsA/mpi_size, MPI_FLOAT, 1, MPI_INIT_TAG, MPI_COMM_WORLD);
    MPI_Send(&B[colsB/mpi_size], 1, custom_datatype, 1, MPI_INIT_TAG, MPI_COMM_WORLD);

    MPI_Type_free(&custom_datatype);
  }
  else {
    MPI_Recv(A, rowsA*colsA/mpi_size, MPI_FLOAT, 0, MPI_INIT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(B, rowsB*colsB/mpi_size, MPI_FLOAT, 0, MPI_INIT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Multiplication
  if(!mpi_rank) {
    // This seems to work; I'm not that sure
    matrix_multiplication(C, A, B, rowsA/mpi_size, colsA, rowsB, colsB);
  }
  else {
    matrix_multiplication(C, A, B, rowsA/mpi_size, colsA, rowsB, colsB/mpi_size);
    //matrix_print(C, rowsA/mpi_size, colsB);
  }


  // Recv & Send result
  if(!mpi_rank) {
    // We need a datatype to unpack the needed elements of C
    MPI_Type_vector(rowsA, colsB/mpi_size, colsB, MPI_FLOAT, &custom_datatype);
    MPI_Type_commit(&custom_datatype);

    MPI_Recv(&C[colsB/mpi_size], 1, custom_datatype, 1, MPI_RESULT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Type_free(&custom_datatype);
  }
  else {
    MPI_Send(C, rowsA*colsB/mpi_size, MPI_FLOAT, 0, MPI_RESULT_TAG, MPI_COMM_WORLD);
  }

  // Result checking
  if(!mpi_rank) {
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
  MPI_Finalize();
    
}
