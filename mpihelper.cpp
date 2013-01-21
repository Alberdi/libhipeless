#include "libhipeless.h"

#include<stdio.h>

int main(int argc, char* argv[]) {
  unsigned int flags = USE_MPI | NON_MPI_ROOT;

  blas_sgemm(NULL, NULL, 0, NULL, NULL, 0, NULL, flags);
  //matrix_multiplication(NULL, NULL, NULL, 0, 0, 0, 0, flags);
}
