#include "libhipeless.h"

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
  int i, j;
  int rowsA, colsA, rowsB, colsB;
  float_matrix A, B, C; 

  int max_size = 64;
  srand((unsigned)time(NULL));
  A.size1 = (int)(rand()%max_size)+1;
  A.size2 = (int)(rand()%max_size)+1;
  B.size2 = (int)(rand()%max_size)+1;

  A.size1 = 2000;
  A.size2 = 2000;
  B.size2 = 2000;

  B.size1 = A.size2;
  C.size1 = A.size1;
  C.size2 = B.size2;

  A.data = (cl_float *) malloc(rowsA*colsA*sizeof(cl_float));
  B.data = (cl_float *) malloc(rowsB*colsB*sizeof(cl_float));
  C.data = (cl_float *) malloc(rowsA*colsB*sizeof(cl_float));

  PM printf("#name:A\n#type:matrix\n#rows:%i\n#columns:%i\n", A.size1, A.size2);
  for(i=0;i<A.size1;i++) {
    for(j=0;j<A.size2;j++) {
      A.data[i*A.size1+j] = (float)(rand() % 256);
      PM printf("%f ", A.data[i*A.size2+j]);
    }
    PM printf("\n");
  }

  PM printf("#name:B\n#type:matrix\n#rows:%i\n#columns:%i\n", B.size1, B.size2);
  for(i=0;i<B.size1;i++) {
    for(j=0;j<B.size2;j++) {
      B.data[i*B.size2+j] = (float)(rand() % 256);
      PM printf("%f ", B.data[i*B.size2+j]);
    }
    PM printf("\n");
  }

printf("PRE\n");
  blas_sgemm(NULL, NULL, 1, &A, &B, 0, &C, flags);
  //matrix_multiplication(C, A, B, rowsA, colsA, rowsB, colsB, flags);
printf("POST\n");

  // Result printing
  PM printf("#name:C\n#type:matrix\n#rows:%i\n#columns:%i\n", C.size1, C.size2);
  float x = 0.0;
  for(i=0; i<C.size1; i++) {
    for(j=0; j<C.size2; j++) {
      x += C.data[i*C.size2+j];
      PM printf("%f ", C.data[i*C.size2+j]);
    }   
    PM printf("\n");
  }

}
