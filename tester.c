#include "libhipeless.h"
#include "minunit.h"

int tests_run = 0;

int equal_matrices(int rows, int cols, float* a, int lda, float* b, int ldb) {
  int i, j;
  float x, y;

  if(lda < cols || ldb < cols) {
    // Wrong parameters, return false
    return 0;
  }

  for(i = 0; i < rows; i++) {
    for(j = 0; j < cols; j++) {
      x = a[i*lda+j];
      y = b[i*ldb+j];
      if(x+y != 0 && fabsf((x-y)/((x+y)/2)) > 0.0001) {
        return 0;
      }
    }
  }
  return 1;
}

void load_matrix(FILE* f, float** a) {
  int rows, cols;
  int i, j;

  fscanf(f, "# name: %*c\n");
  fscanf(f, "# type: matrix\n");
  fscanf(f, "# rows: %i\n", &rows);
  fscanf(f, "# columns: %i\n", &cols);

  *a = (float*) malloc(rows*cols*sizeof(float));
  
  for(i = 0; i < rows; i++) {
    for(j = 0; j < cols; j++) {
      fscanf(f, "%f ", &(*a)[i*rows+j]);
    }
  }
}

void load_file(const char* filename, float** a, float** b, float** c) {
  FILE* f;
  f = fopen(filename, "r");
  load_matrix(f, a);
  load_matrix(f, b);
  if(c != NULL) {
    load_matrix(f, c);
  }
  fclose(f);
}

static const char* test_tester() {
  float *a, *b;
  load_file("tests/xgemm_ones.txt", &a, &b, NULL);

  // A = B = ones(32, 32);
  mu_assert("Error in test_tester(0).", equal_matrices(32, 32, a, 32, b, 32));

  a[15] = 1.000001;
  // A ~= B
  mu_assert("Error in test_tester(1).", equal_matrices(32, 32, a, 32, b, 32));

  a[15] = 2;
  // A != B
  mu_assert("Error in test_tester(2).", !equal_matrices(32, 32, a, 32, b, 32));

  // Wrong parameters, should be false
  mu_assert("Error in test_tester(3).", !equal_matrices(32, 34, a, 32, b, 32));
  mu_assert("Error in test_tester(4).", !equal_matrices(32, 32, a, 32, b, 30));

  free(a);
  free(b);
  return 0;
}

static const char* test_sgemm_ones() {
  float *a, *b, *c, *d;
  load_file("tests/xgemm_ones.txt", &a, &b, &c);

  d = (float*) malloc(32*32*sizeof(float));
  
  // D == C
  blas_sgemm('N', 'N', 32, 32, 32, 1, a, 32, b, 32, 0, d, 32, USE_CPU);
  mu_assert("Error in test_sgemm_ones(0).", equal_matrices(32, 32, d, 32, c, 32));

  // D == C when op(A) == A', (A is symmetric).
  blas_sgemm('T', 'N', 32, 32, 32, 1, a, 32, b, 32, 0, d, 32, USE_CPU);
  mu_assert("Error in test_sgemm_ones(1).", equal_matrices(32, 32, d, 32, c, 32));

  // D == C when op(A) == A', (A is symmetric).
  blas_sgemm('T', 'N', 32, 32, 32, 1, a, 32, b, 32, 0, d, 32, USE_CPU);
  mu_assert("Error in test_sgemm_ones(2).", equal_matrices(32, 32, d, 32, c, 32));

  // D == C when op(B) == B' (B is symmetric).
  blas_sgemm('N', 'T', 32, 32, 32, 1, a, 32, b, 32, 0, d, 32, USE_CPU);
  mu_assert("Error in test_sgemm_ones(2).", equal_matrices(32, 32, d, 32, c, 32));
  
  // D == C when op(B) == B', op(A) == A' (A, B are symmetric).
  blas_sgemm('T', 'T', 32, 32, 32, 1, a, 32, b, 32, 0, d, 32, USE_CPU);
  mu_assert("Error in test_sgemm_ones(2).", equal_matrices(32, 32, d, 32, c, 32));

  free(a); free(b); free(c); free(d);
  return 0;
}

static const char* all_tests() {
  mu_run_test(test_tester);
  mu_run_test(test_sgemm_ones);
  return 0;
}

int main(int argc, char* argv[]) {
  const char* result = all_tests();
  if(result != 0) {
    printf("%s\n", result);
  }
  else {
    printf("ALL TESTS PASSED\n");
  }

  return result != 0;
}
