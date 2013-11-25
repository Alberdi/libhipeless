#include "libhipeless.h"
#include "minunit.h"

int tests_run = 0;

template <typename number>
int equal_matrices(int rows, int cols, number* a, int lda, number* b, int ldb) {
  int i, j;
  number x, y;

  if(lda < cols || ldb < cols) {
    // Wrong parameters, return false
    return 0;
  }

  for(i = 0; i < rows; i++) {
    for(j = 0; j < cols; j++) {
      x = a[i*lda+j];
      y = b[i*ldb+j];
      if(x+y != 0 && fabs((x-y)/((x+y)/2)) > 0.0001) {
        return 0;
      }
    }
  }
  return 1;
}

template <typename number>
void load_matrix(FILE* f, number** a) {
  int rows, cols;
  int i, j;

  fscanf(f, "# name: %*c\n");
  fscanf(f, "# type: matrix\n");
  fscanf(f, "# rows: %i\n", &rows);
  fscanf(f, "# columns: %i\n", &cols);

  *a = (number*) malloc(rows*cols*sizeof(number));
  
  for(i = 0; i < rows; i++) {
    for(j = 0; j < cols; j++) {
      fscanf(f, sizeof(number) == sizeof(float) ? "%f " : "%lf ", &(*a)[i*cols+j]);
    }
  }
}

template <typename number>
void load_file(const char* filename, number** a, number** b, number** c) {
  FILE* f;
  f = fopen(filename, "r");
  load_matrix(f, a);
  load_matrix(f, b);
  if(c != NULL) {
    load_matrix(f, c);
  }
  fclose(f);
}

template <typename number>
static const char* test_tester(int flags, number t) {
  number *a, *b, *c;
  c = NULL;
  load_file("tests/xgemm_ones.txt", &a, &b, &c);

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

template <typename number>
static const char* test_xgemm_ones(int flags, number t) {
  number *a, *b, *c, *d;
  load_file("tests/xgemm_ones.txt", &a, &b, &c);

  d = (number*) malloc(32*32*sizeof(number));
  
  // D == C
  blas_xgemm('N', 'N', 32, 32, 32, (number)(number)1, a, 32, b, 32, (number)(number)0, d, 32, flags);
  mu_assert("Error in test_xgemm_ones(0).", equal_matrices(32, 32, d, 32, c, 32));

  // D == C when op(A) == A', (A is symmetric).
  blas_xgemm('T', 'N', 32, 32, 32, (number)(number)1, a, 32, b, 32, (number)(number)0, d, 32, flags);
  mu_assert("Error in test_xgemm_ones(1).", equal_matrices(32, 32, d, 32, c, 32));

  // D == C when op(A) == A', (A is symmetric).
  blas_xgemm('T', 'N', 32, 32, 32, (number)(number)1, a, 32, b, 32, (number)(number)0, d, 32, flags);
  mu_assert("Error in test_xgemm_ones(2).", equal_matrices(32, 32, d, 32, c, 32));

  // D == C when op(B) == B' (B is symmetric).
  blas_xgemm('N', 'T', 32, 32, 32, (number)(number)1, a, 32, b, 32, (number)(number)0, d, 32, flags);
  mu_assert("Error in test_xgemm_ones(3).", equal_matrices(32, 32, d, 32, c, 32));
  
  // D == C when op(B) == B', op(A) == A' (A, B are symmetric).
  blas_xgemm('T', 'T', 32, 32, 32, (number)(number)1, a, 32, b, 32, (number)(number)0, d, 32, flags);
  mu_assert("Error in test_xgemm_ones(4).", equal_matrices(32, 32, d, 32, c, 32));

  free(a); free(b); free(c); free(d);
  return 0;
}

template <typename number>
static const char* test_xgemm_rand(int flags, number t) {
  number *a, *b, *c, *d;
  load_file("tests/xgemm_rand.txt", &a, &b, &c);

  d = (number*) malloc(32*32*sizeof(number));

  // D == C
  blas_xgemm('N', 'N', 32, 32, 32, (number)1, a, 32, b, 32, (number)0, d, 32, flags);
  mu_assert("Error in test_xgemm_rand(0).", equal_matrices(32, 32, d, 32, c, 32));

  // D != C when op(A) == A', (A is not symmetric).
  blas_xgemm('T', 'N', 32, 32, 32, (number)1, a, 32, b, 32, (number)0, d, 32, flags);
  mu_assert("Error in test_xgemm_rand(1).", !equal_matrices(32, 32, d, 32, c, 32));

  // D != C when op(A) == A', (A is not symmetric).
  blas_xgemm('T', 'N', 32, 32, 32, (number)1, a, 32, b, 32, (number)0, d, 32, flags);
  mu_assert("Error in test_xgemm_rand(2).", !equal_matrices(32, 32, d, 32, c, 32));

  // D != C when op(B) == B' (B is not symmetric).
  blas_xgemm('N', 'T', 32, 32, 32, (number)1, a, 32, b, 32, (number)0, d, 32, flags);
  mu_assert("Error in test_xgemm_rand(3).", !equal_matrices(32, 32, d, 32, c, 32));
  
  // D != C when op(B) == B', op(A) == A' (A, B are not symmetric).
  blas_xgemm('T', 'T', 32, 32, 32, (number)1, a, 32, b, 32, (number)0, d, 32, flags);
  mu_assert("Error in test_xgemm_rand(4).", !equal_matrices(32, 32, d, 32, c, 32));

  free(a); free(b); free(c); free(d);
  return 0;
}

template <typename number>
static const char* test_xgemm_rand_big(int flags, number t) {
  number *a, *b, *c, *d;
  // A = 331x137; B = 137x401; C = 331x401;
  load_file("tests/xgemm_rand_big.txt", &a, &b, &c);

  d = (number*) malloc(331*405*sizeof(number));

  // D == C
  blas_xgemm('N', 'N', 331, 401, 137, (number)1, a, 137, b, 401, (number)0, d, 405, flags);
  mu_assert("Error in test_xgemm_rand_big(0).", equal_matrices(331, 401, d, 405, c, 401));

  // D == C (first 15 rows)
  blas_xgemm('N', 'N', 15, 401, 137, (number)1, a, 137, b, 401, (number)0, d, 405, flags);
  mu_assert("Error in test_xgemm_rand_big(1).", equal_matrices(15, 401, d, 405, c, 401));

  // D != C (first 15 rows, using 15 columns of C)
  blas_xgemm('N', 'N', 4, 4, 137, (number)1, a, 137, b, 401, (number)0, d, 405, flags);
  mu_assert("Error in test_xgemm_rand_big(2).", equal_matrices(15, 15, d, 405, c, 401));

  free(a); free(b); free(c); free(d);
  return 0;
}

template <typename number>
static const char* test_xgemm_rand_big_alphabeta(int flags, number t) {
  number *a, *b, *c, *d;
  int i, j;
  // A = 331x137; B = 137x401; C = 331x401;
  // alpha = 12.345; C = alpha*A*B-210;
  load_file("tests/xgemm_rand_big_alphabeta.txt", &a, &b, &c);

  d = (number*) malloc(331*401*sizeof(number));
  for(i = 0; i < 331; i++) {
    for(j = 0; j < 401; j++) {
      d[i*401+j] = 100;
    }
  }

  // D == C (beta = -2)
  blas_xgemm('N', 'N', 331, 401, 137, (number)12.345, a, 137, b, 401, (number)-2.1, d, 401, flags);
  mu_assert("Error in test_xgemm_rand_big_alphabeta(0).", equal_matrices(331, 401, d, 401, c, 401));

  free(a); free(b); free(c); free(d);
  return 0;
}

template <typename number>
static const char* test_xgemm_row(int flags, number t) {
  number *a, *b, *c, *d;
  load_file("tests/xgemm_row.txt", &a, &b, &c);

  d = (number*) malloc(1*1*sizeof(number));

  // D == C
  blas_xgemm('N', 'N', 1, 1, 128, (number)1, a, 128, b, 1, (number)0, d, 1, flags);
  mu_assert("Error in test_xgemm_row(0).", equal_matrices(1, 1, d, 1, c, 1));

  // D == 12.34*C (alpha = 12.34)
  blas_xgemm('N', 'N', 1, 1, 128, (number)12.34, a, 128, b, 1, (number)0, d, 1, flags);
  d[0] /= 12.34;
  mu_assert("Error in test_xgemm_row(1).", equal_matrices(1, 1, d, 1, c, 1));

  // We'll store in D the calculation of the first 4 multiplications.
  blas_xgemm('N', 'N', 1, 1, 4, (number)1, a, 128, b, 1, (number)0, d, 1, flags);
  // And now we replicate it in C.
  c[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
  mu_assert("Error in test_xgemm_row(2).", equal_matrices(1, 1, d, 1, c, 1));

  // Now, we'll use pointers to calculate the following 4 multiplications.
  blas_xgemm('N', 'N', 1, 1, 4, (number)1, &a[4], 124, &b[4], 1, (number)0, d, 1, flags);
  // And now we replicate it in C.
  c[0] = a[4]*b[4] + a[5]*b[5] + a[6]*b[6] + a[7]*b[7];
  mu_assert("Error in test_xgemm_row(3).", equal_matrices(1, 1, d, 1, c, 1));

  // Do the same with alpha = 1.2 and beta = 2.3
  blas_xgemm('N', 'N', 1, 1, 4, (number)1.2, &a[4], 124, &b[4], 1, (number)2.3, d, 1, flags);
  // And now we replicate it in C.
  // C = 1.2*C + 2.3*C = 3.5*C
  c[0] = 3.5*c[0];
  mu_assert("Error in test_xgemm_row(4).", equal_matrices(1, 1, d, 1, c, 1));

  free(a); free(b); free(c); free(d);
  return 0;
}

template <typename number>
static const char* test_xgemm_row_trans(int flags, number t) {
  number *a, *b, *c, *d;
  load_file("tests/xgemm_row_trans.txt", &a, &b, &c);

  d = (number*) malloc(128*128*sizeof(number));

  // D == C
  blas_xgemm('T', 'T', 128, 128, 1, (number)1, a, 128, b, 1, (number)0, d, 128, flags);
  mu_assert("Error in test_xgemm_row_trans(0).", equal_matrices(128, 128, d, 128, c, 128));

  // D[0:3] == C[0:3]
  blas_xgemm('T', 'T', 4, 4, 1, (number)1, a, 128, b, 1, (number)0, d, 128, flags);
  mu_assert("Error in test_xgemm_row_trans(1).", equal_matrices(4, 4, d, 128, c, 128));

  // Let's be sure we're not overflowing d when calculating a fraction.
  free(d);
  d = (number*) malloc(35*35*sizeof(number));
  // D == C (35 rows, 35 columns).
  blas_xgemm('T', 'T', 35, 35, 1, (number)1, a, 128, b, 1, (number)0, d, 35, flags);
  mu_assert("Error in test_xgemm_row_trans(2).", equal_matrices(35, 35, d, 35, c, 128));

  free(a); free(b); free(c); free(d);
  return 0;
}

template <typename number>
static const char* test_xtrmm_ones_llnx(int flags, number t) {
  // Left, lower, not transposed
  number *a, *b, *c;
  load_file("tests/xtrmm_ones_llnx.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('L', 'L', 'N', 'N', 32, 32, (number)1, a, 32, b, 32, flags);
  mu_assert("Error in test_xtrmm_ones_llnx(0).", equal_matrices(32, 32, b, 32, c, 32));

  load_file("tests/xtrmm_ones_llnx.txt", &a, &b, &c);

  // B == C (when diagonal of A is treated as ones)
  blas_xtrmm('L', 'L', 'N', 'N', 32, 32, (number)1, a, 32, b, 32, flags);
  mu_assert("Error in test_xtrmm_ones_llnx(1).", equal_matrices(32, 32, b, 32, c, 32));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_ones_lltx(int flags, number t) {
  // Left, lower, transposed
  number *a, *b, *c;
  load_file("tests/xtrmm_ones_lltx.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('L', 'L', 'T', 'N', 32, 32, (number)1, a, 32, b, 32, flags);
  mu_assert("Error in test_xtrmm_ones_lltx(0).", equal_matrices(32, 32, b, 32, c, 32));

  load_file("tests/xtrmm_ones_lltx.txt", &a, &b, &c);

  // B == C (when diagonal of A is treated as ones)
  blas_xtrmm('L', 'L', 'T', 'N', 32, 32, (number)1, a, 32, b, 32, flags);
  mu_assert("Error in test_xtrmm_ones_lltx(1).", equal_matrices(32, 32, b, 32, c, 32));

  free(a); free(b); free(c);
  return 0;
}
template <typename number>
static const char* all_tests(number t) {
  int i;
  int flags[4] = {USE_CPU, USE_GPU, USE_CPU | USE_MPI, USE_GPU | USE_MPI};

  tests_run = 0;
  mu_run_test(test_tester, flags[0], t);
  printf("Tester ok.\n");

  for(i = 0; i < 4; i++) {
    tests_run = 0;
    printf("Using flags = 0x%x.\n", flags[i], t);
    mu_run_test(test_xgemm_ones, flags[i], t);
    mu_run_test(test_xgemm_rand, flags[i], t);
    mu_run_test(test_xgemm_rand_big, flags[i], t);
    mu_run_test(test_xgemm_rand_big_alphabeta, flags[i], t);
    mu_run_test(test_xgemm_row, flags[i], t);
    mu_run_test(test_xgemm_row_trans, flags[i], t);
    mu_run_test(test_xtrmm_ones_llnx, flags[i], t);
    mu_run_test(test_xtrmm_ones_lltx, flags[i], t);
  }
  return 0;
}

static const char* all_tests() {
  const char* result;
  printf("TESTING FLOATS\n");
  result = all_tests((float) 1);
  if(result != 0) {
    return result;
  }
  printf("\nTESTING DOUBLES\n");
  result = all_tests((double) 1);
  return result;
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
