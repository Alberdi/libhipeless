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
  unsigned int flags = USE_CPU | USE_MPI;
  cl_int i, j, m, k, n;
  cl_int lda, ldb, ldc;
  cl_float *a, *b, *c;
  cl_float alpha, beta;
  cl_char transa, transb;
  int rowsa, colsa, rowsb, colsb;

  if(flags & USE_MPI) {
    MPI_Init(&argc, &argv);
  }

  int max_size = 14;
  srand((unsigned)time(NULL));
  m = (int)(rand()%max_size)+16;
  k = (int)(rand()%max_size)+16;
  n = (int)(rand()%max_size)+16;

  transa = 'N';
  if(transa == 'N') {
    rowsa = m;
    colsa = k;
  }
  else {
    rowsa = k;
    colsa = m;
  }
  lda = colsa+(rand()%max_size)+1;

  transb = 'T';
  if(transb == 'N') {
    rowsb = k;
    colsb = n;
  }
  else {
    rowsb = n;
    colsb = k;
  }
  ldb = colsb+(rand()%max_size)+1;
  ldc = n+(rand()%max_size)+1;

  rowsa = 64;
  colsa = 64;
  lda = 64;

  rowsb = 64;
  colsb = 64;
  ldb = 64;

  a = (cl_float *) malloc(rowsa*lda*sizeof(cl_float));
  b = (cl_float *) malloc(rowsb*ldb*sizeof(cl_float));
  c = (cl_float *) malloc(m*ldc*sizeof(cl_float));

  PM printf("#name:A\n#type:matrix\n#rows:%i\n#columns:%i\n", rowsa, colsa);
  for(i=0; i<rowsa; i++) {
    for(j=0; j<colsa; j++) {
      //a[i*lda+j] = (float)(rand() % 256);
      if(j < i) {
        a[i*lda+j] = 0;
      }
      else {
        a[i*lda+j] = j+1;
      }
      PM printf("%.0f ", a[i*lda+j]);
    }
    PM printf("\n");
  }

  PM printf("#name:B\n#type:matrix\n#rows:%i\n#columns:%i\n", rowsb, colsb);
  for(i=0; i<rowsb; i++) {
    for(j=0; j<colsb; j++) {
      b[i*ldb+j] = (float)(rand() % 6);
      PM printf("%.0f ", b[i*ldb+j]);
    }
    PM printf("\n");
  }

  for(i=0; i<m; i++) {
    for(j=0; j<n ;j++) {
      c[i*ldc+j] = 10000;
    }
  }

  alpha = 1;
  beta = 1.5;
//  blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, flags);
  blas_strmm('L', 'U', 'N', 'N', rowsb, colsb, alpha, a, lda, b, ldb, flags);

  PM printf("#name:C\n#type:matrix\n#rows:%i\n#columns:%i\n", rowsb, colsb);
  for(i=0; i<rowsb; i++) {
    for(j=0; j<colsb; j++) {
      PM printf("%.0f ", b[i*ldb+j]);
    }
    PM printf("\n");
  }

/*  // Result printing
  PM printf("#name:C\n#type:matrix\n#rows:%i\n#columns:%i\n", m, n);
  for(i=0; i<m; i++) {
    for(j=0; j<n; j++) {
      PM printf("%.0f ", c[i*ldc+j]);
    }
    PM printf("\n");
  }
*/
  if(flags & USE_MPI) {
    MPI_Finalize();
  }
}
