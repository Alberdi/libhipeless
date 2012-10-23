#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "libhipeless.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
  unsigned int flags = USE_GPU | USE_MPI;
  int i, j;
  int rowsA = 1024, colsA = 512, rowsB = 512, colsB = 2048;
  cl_float *A, *B, *C; 

  A = (cl_float *) malloc(rowsA*colsA*sizeof(cl_float));
  B = (cl_float *) malloc(rowsB*colsB*sizeof(cl_float));
  C = (cl_float *) malloc(rowsA*colsB*sizeof(cl_float));

  for(i=0;i<rowsA;i++)
    for(j=0;j<colsA;j++)
      A[i*colsA+j]=1;

  for(i=0;i<rowsB;i++)
    for(j=0;j<colsB;j++)
      B[i*colsB+j] = i==j ? 1 : 0;

  matrix_multiplication(C, A, B, rowsA, colsA, rowsB, colsB, flags, argc, argv);

  // Result checking
  float x = 0.0;
  for(i=0; i<rowsA; i++) {
    for(j=0; j<colsB; j++) {
      x += C[i*colsB+j];
    }   
  }

  // TODO This check is not correct always
  if(x==rowsA*colsA || x==rowsA*colsB) printf("CORRECTO (%f)\n", x);
  else printf("INCORRECTO: %f (%d, %d)\n", x, rowsA*colsA, rowsA*colsB);

  //matrix_print(C, rowsA, colsB);

}
