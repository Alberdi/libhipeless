#include "libhipeless.h"

#include <mpi.h>

int main(int argc, char* argv[]) {
  unsigned int flags = USE_MPI | NON_MPI_ROOT;
  int function;

  MPI_Init(&argc, &argv);

  MPI_Comm parent;
  MPI_Comm_get_parent(&parent);

  MPI_Bcast(&function, 1, MPI_INTEGER, 0, parent);

  if(function == SGEMM) {
    blas_sgemm('0', '0', 0, 0, 0, 0, NULL, 0, NULL, 0, 0, NULL, 0, flags);
  }
  else if (function == DGEMM) {
    blas_dgemm('0', '0', 0, 0, 0, 0, NULL, 0, NULL, 0, 0, NULL, 0, flags);
  }
  else if(function == STRMM) {
    blas_strmm('0', '0', '0', '0', 0, 0, 0, NULL, 0, NULL, 0, flags);
  }
  else if(function == DTRMM) {
    blas_dtrmm('0', '0', '0', '0', 0, 0, 0, NULL, 0, NULL, 0, flags);
  }


  MPI_Finalize();
}
