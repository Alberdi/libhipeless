#include <CL/cl.h>

#define USE_CPU 0x01
#define USE_GPU 0x02
#define USE_MPI 0x04

int matrix_multiplication(cl_float *C, cl_float *A, cl_float *B, cl_uint rowsA, cl_uint colsA, cl_uint rowsB, cl_uint colsB,
                          unsigned int flags, int argc, char* argv[]);
