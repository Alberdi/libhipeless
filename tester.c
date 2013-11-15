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
      fscanf(f, "%f ", &(*a)[i*cols+j]);
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

static const char* test_tester(int flags) {
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

static const char* test_sgemm_ones(int flags) {
  float *a, *b, *c, *d;
  load_file("tests/xgemm_ones.txt", &a, &b, &c);

  d = (float*) malloc(32*32*sizeof(float));
  
  // D == C
  blas_sgemm('N', 'N', 32, 32, 32, 1, a, 32, b, 32, 0, d, 32, flags);
  mu_assert("Error in test_sgemm_ones(0).", equal_matrices(32, 32, d, 32, c, 32));

  // D == C when op(A) == A', (A is symmetric).
  blas_sgemm('T', 'N', 32, 32, 32, 1, a, 32, b, 32, 0, d, 32, flags);
  mu_assert("Error in test_sgemm_ones(1).", equal_matrices(32, 32, d, 32, c, 32));

  // D == C when op(A) == A', (A is symmetric).
  blas_sgemm('T', 'N', 32, 32, 32, 1, a, 32, b, 32, 0, d, 32, flags);
  mu_assert("Error in test_sgemm_ones(2).", equal_matrices(32, 32, d, 32, c, 32));

  // D == C when op(B) == B' (B is symmetric).
  blas_sgemm('N', 'T', 32, 32, 32, 1, a, 32, b, 32, 0, d, 32, flags);
  mu_assert("Error in test_sgemm_ones(3).", equal_matrices(32, 32, d, 32, c, 32));
  
  // D == C when op(B) == B', op(A) == A' (A, B are symmetric).
  blas_sgemm('T', 'T', 32, 32, 32, 1, a, 32, b, 32, 0, d, 32, flags);
  mu_assert("Error in test_sgemm_ones(4).", equal_matrices(32, 32, d, 32, c, 32));

  free(a); free(b); free(c); free(d);
  return 0;
}

static const char* test_sgemm_rand(int flags) {
  float *a, *b, *c, *d;
  load_file("tests/xgemm_rand.txt", &a, &b, &c);

  d = (float*) malloc(32*32*sizeof(float));

  // D == C
  blas_sgemm('N', 'N', 32, 32, 32, 1, a, 32, b, 32, 0, d, 32, flags);
  mu_assert("Error in test_sgemm_ones(0).", equal_matrices(32, 32, d, 32, c, 32));

  // D != C when op(A) == A', (A is not symmetric).
  blas_sgemm('T', 'N', 32, 32, 32, 1, a, 32, b, 32, 0, d, 32, flags);
  mu_assert("Error in test_sgemm_rand(1).", !equal_matrices(32, 32, d, 32, c, 32));

  // D != C when op(A) == A', (A is not symmetric).
  blas_sgemm('T', 'N', 32, 32, 32, 1, a, 32, b, 32, 0, d, 32, flags);
  mu_assert("Error in test_sgemm_rand(2).", !equal_matrices(32, 32, d, 32, c, 32));

  // D != C when op(B) == B' (B is not symmetric).
  blas_sgemm('N', 'T', 32, 32, 32, 1, a, 32, b, 32, 0, d, 32, flags);
  mu_assert("Error in test_sgemm_rand(3).", !equal_matrices(32, 32, d, 32, c, 32));
  
  // D != C when op(B) == B', op(A) == A' (A, B are not symmetric).
  blas_sgemm('T', 'T', 32, 32, 32, 1, a, 32, b, 32, 0, d, 32, flags);
  mu_assert("Error in test_sgemm_rand(4).", !equal_matrices(32, 32, d, 32, c, 32));

  free(a); free(b); free(c); free(d);
  return 0;
}

static const char* test_sgemm_row(int flags) {
  float *a, *b, *c, *d;
  load_file("tests/xgemm_row.txt", &a, &b, &c);

  d = (float*) malloc(1*1*sizeof(float));

  // D == C
  blas_sgemm('N', 'N', 1, 1, 128, 1, a, 128, b, 1, 0, d, 1, flags);
  mu_assert("Error in test_sgemm_row(0).", equal_matrices(1, 1, d, 1, c, 1));

  free(a); free(b); free(c); free(d);
  return 0;
}

static const char* all_tests() {
  int i;
  int flags[4] = {USE_CPU, USE_GPU, USE_CPU | USE_MPI, USE_GPU | USE_MPI};
  for(i = 0; i < 4; i++) {
    tests_run = 0;
    printf("Using flags = 0x%x.\n", flags[i]);
    mu_run_test(test_tester, flags[i]);
    mu_run_test(test_sgemm_ones, flags[i]);
    mu_run_test(test_sgemm_rand, flags[i]);
    mu_run_test(test_sgemm_row, flags[i]);
  }
  return 0;
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  const char* result = all_tests();
  if(result != 0) {
    printf("%s\n", result);
  }
  else {
    printf("ALL TESTS PASSED\n");
  }

  MPI_Finalize();
  return result != 0;
}
