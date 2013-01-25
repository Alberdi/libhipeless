#include "libhipeless.h"

#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
  unsigned int flags = USE_MPI | NON_MPI_ROOT;

  MPI_Init(&argc, &argv);

  blas_sgemm(NULL, NULL, 0, NULL, NULL, 0, NULL, flags);

  MPI_Finalize();
}
