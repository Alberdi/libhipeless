#include "libhipeless.h"

#include <stdio.h>
#include <stdlib.h>

#define PRINT_MATRICES 1

#ifdef PRINT_MATRICES
  #define PM if(1)
#else
  #define PM if(0)
#endif

int main(int argc, char* argv[]) {
  unsigned int flags = USE_GPU;
  int i, j;
  int rowsA = 1024, colsA = 512, rowsB = 512, colsB = 2048;
  //int rowsA = 64, colsA = 32, rowsB = 32, colsB = 32;
  cl_float *A, *B, *C; 

  A = (cl_float *) malloc(rowsA*colsA*sizeof(cl_float));
  B = (cl_float *) malloc(rowsB*colsB*sizeof(cl_float));
  C = (cl_float *) malloc(rowsA*colsB*sizeof(cl_float));

  PM printf("#name:A\n#type:matrix\n#rows:%i\n#columns:%i\n", rowsA, colsA);
  for(i=0;i<rowsA;i++) {
    for(j=0;j<colsA;j++) {
      A[i*colsA+j] = (float)rand();
      PM printf("%f ", A[i*colsA+j]);
    }
    PM printf("\n");
  }

  PM printf("#name:B\n#type:matrix\n#rows:%i\n#columns:%i\n", rowsB, colsB);
  for(i=0;i<rowsB;i++) {
    for(j=0;j<colsB;j++) {
      B[i*colsB+j] = (float)rand();
      PM printf("%f ", B[i*colsB+j]);
    }
    PM printf("\n");
  }

  matrix_multiplication(C, A, B, rowsA, colsA, rowsB, colsB, flags);

  // Result checking
  PM printf("#name:C\n#type:matrix\n#rows:%i\n#columns:%i\n", rowsA, colsB);
  float x = 0.0;
  for(i=0; i<rowsA; i++) {
    for(j=0; j<colsB; j++) {
      x += C[i*colsB+j];
      PM printf("%f ", C[i*colsB+j]);
    }   
    PM printf("\n");
  }

  // TODO This check is not correct always
  //if(x==rowsA*colsA || x==rowsA*colsB) printf("CORRECTO (%f)\n", x);
  //else printf("INCORRECTO: %f (%d, %d)\n", x, rowsA*colsA, rowsA*colsB);

  //matrix_print(C, rowsA, colsB);

}
