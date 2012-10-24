#include "libhipeless.h"

#include<stdio.h>

int main(int argc, char* argv[]) {
  cl_float *A, *B, *C;
  int rowsA = 1024, colsA = 512, rowsB = 512, colsB = 2048;
  unsigned int flags = USE_GPU | USE_MPI | NON_MPI_ROOT;

  matrix_multiplication(C, A, B, rowsA, colsA, rowsB, colsB, flags, argc, argv);
}
