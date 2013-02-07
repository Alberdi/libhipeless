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
  cl_char transa;
  int rowsa, colsa;

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

  transa = 'T';
  if(transa == 'N') {
    rowsa = m;
    colsa = k;
  }
  else {
    rowsa = k;
    colsa = m;
  }

  PM printf("#name:A\n#type:matrix\n#rows:%i\n#columns:%i\n", rowsa, colsa);
  for(i=0; i<rowsa; i++) {
    for(j=0; j<colsa; j++) {
      a[i*colsa+j] = 1;//(float)(rand() % 256);
      PM printf("%f ", a[i*colsa+j]);
    }
    PM printf("\n");
  }

  PM printf("#name:B\n#type:matrix\n#rows:%i\n#columns:%i\n", k, n);
  for(i=0; i<k; i++) {
    for(j=0; j<n; j++) {
      b[i*n+j] = 1;//(float)(rand() % 256);
      PM printf("%f ", b[i*n+j]);
    }
    PM printf("\n");
  }

  for(i=0; i<m; i++) {
    for(j=0; j<n ;j++) {
      c[i*n+j] = 10000;
    }
  }

  blas_sgemm(transa, 'N', m, n, k, 1, a, m, b, k, 0, c, m, flags);

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
