#include "libhipeless.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PRINT_MATRICES 1

#ifdef PRINT_MATRICES
  #define PM if(1)
#else
  #define PM if(0)
#endif

int main(int argc, char* argv[]) {
  unsigned int flags = USE_CPU;
  cl_int i, j, m, k, n;
  cl_float *a, *b, *c;

  if(flags & USE_MPI) {
    MPI_Init(&argc, &argv);
  }

  int max_size = 64;
  srand((unsigned)time(NULL));
  m = (int)(rand()%max_size)+1;
  k = (int)(rand()%max_size)+1;
  n = (int)(rand()%max_size)+1;

  a = (cl_float *) malloc(m*k*sizeof(cl_float));
  b = (cl_float *) malloc(k*n*sizeof(cl_float));
  c = (cl_float *) malloc(m*n*sizeof(cl_float));

  PM printf("#name:A\n#type:matrix\n#rows:%i\n#columns:%i\n", m, k);
  for(i=0; i<m; i++) {
    for(j=0; j<k; j++) {
      a[i*k+j] = (float)(rand() % 256);
      PM printf("%f ", a[i*k+j]);
    }
    PM printf("\n");
  }

  PM printf("#name:B\n#type:matrix\n#rows:%i\n#columns:%i\n", k, n);
  for(i=0; i<k; i++) {
    for(j=0; j<n; j++) {
      b[i*n+j] = (float)(rand() % 256);
      PM printf("%f ", b[i*n+j]);
    }
    PM printf("\n");
  }

  for(i=0; i<m; i++) {
    for(j=0; j<n ;j++) {
      c[i*n+j] = 10000;
    }
  }

  blas_sgemm('N', 'N', m, n, k, 1, a, m, b, k, 0, c, m, flags);

  // Result printing
  PM printf("#name:C\n#type:matrix\n#rows:%i\n#columns:%i\n", m, n);
  for(i=0; i<m; i++) {
    for(j=0; j<n; j++) {
      PM printf("%f ", c[i*n+j]);
    }
    PM printf("\n");
  }

  if(flags & USE_MPI) {
    MPI_Finalize();
  }
}
