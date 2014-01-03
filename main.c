#include "libhipeless.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

//#define PRINT_MATRICES 1

#ifdef PRINT_MATRICES
  #define PM if(1)
#else
  #define PM if(0)
#endif

int main(int argc, char* argv[]) {
  typedef cl_float number;
//  typedef cl_double number;
  unsigned int flags = USE_CPU;
  cl_int i, j, m, k, n;
  cl_int lda, ldb, ldc;
  number *a, *b, *c;
  number alpha, beta;
  cl_char transa, transb;
  int rowsa, colsa, rowsb, colsb;
  timeval t0, t1;
  double elapsed;

  if(flags & USE_MPI) {
    MPI_Init(&argc, &argv);
  }

  int max_size = 14;
  srand((unsigned)time(NULL));
  m = (int)(rand()%max_size)+16;
  k = (int)(rand()%max_size)+16;
  n = (int)(rand()%max_size)+16;
  m = 2048*2;
  n = 2048*2;
  k = 2048*2;

  transa = 'N';
  if(transa == 'N') {
    rowsa = m;
    colsa = k;
  }
  else {
    rowsa = k;
    colsa = m;
  }
  lda = colsa;

  transb = 'N';
  if(transb == 'N') {
    rowsb = k;
    colsb = n;
  }
  else {
    rowsb = n;
    colsb = k;
  }
  ldb = colsb;
  ldc = n;

  a = (number *) malloc(rowsa*lda*sizeof(number));
  b = (number *) malloc(rowsb*ldb*sizeof(number));
  c = (number *) malloc(m*ldc*sizeof(number));

  PM printf("#name:A\n#type:matrix\n#rows:%i\n#columns:%i\n", rowsa, colsa);
  for(i=0; i<rowsa; i++) {
    for(j=0; j<colsa; j++) {
      a[i*lda+j] = (number)(rand() % 256);
      PM printf("%.0f ", a[i*lda+j]);
    }
    PM printf("\n");
  }

  PM printf("#name:B\n#type:matrix\n#rows:%i\n#columns:%i\n", rowsb, colsb);
  for(i=0; i<rowsb; i++) {
    for(j=0; j<colsb; j++) {
      b[i*ldb+j] = (number)(rand() % 256);
      PM printf("%.0f ", b[i*ldb+j]);
    }
    PM printf("\n");
  }

  alpha = 1.0;
  beta = 0.0;

  gettimeofday(&t0, NULL);
  blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, flags);
  //blas_strmm('L', 'U', 'N', 'N', rowsb, colsb, alpha, a, lda, b, ldb, flags);
  gettimeofday(&t1, NULL);

  PM printf("#name:C\n#type:matrix\n#rows:%i\n#columns:%i\n", rowsb, colsb);
  for(i=0; i<rowsb; i++) {
    for(j=0; j<colsb; j++) {
      PM printf("%.0f ", b[i*ldb+j]);
    }
    PM printf("\n");
  }

  elapsed = (t1.tv_sec - t0.tv_sec);
  elapsed += (t1.tv_usec - t0.tv_usec) / 1000000.0;   // usec to seconds.
  printf("Elapsed time: %f seconds.\n", elapsed);

  // Result printing
  PM printf("#name:C\n#type:matrix\n#rows:%i\n#columns:%i\n", m, n);
  for(i=0; i<m; i++) {
    for(j=0; j<n; j++) {
      PM printf("%.0f ", c[i*ldc+j]);
    }
    PM printf("\n");
  }

  if(flags & USE_MPI) {
    MPI_Finalize();
  }
}
