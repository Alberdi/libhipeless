#include "libhipeless.h"

#include<stdio.h>

int main(int argc, char* argv[]) {
  unsigned int flags = USE_MPI | NON_MPI_ROOT;

  matrix_multiplication(NULL, NULL, NULL, 0, 0, 0, 0, flags);
}
