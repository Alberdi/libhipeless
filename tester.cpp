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
        //printf("Discrepancy in %i, %i: %lf - %lf\n", i, j, x, y);
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
  blas_xtrmm('L', 'L', 'N', 'U', 32, 32, (number)1, a, 32, b, 32, flags);
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
  blas_xtrmm('L', 'L', 'T', 'U', 32, 32, (number)1, a, 32, b, 32, flags);
  mu_assert("Error in test_xtrmm_ones_lltx(1).", equal_matrices(32, 32, b, 32, c, 32));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_ones_lunx(int flags, number t) {
  // Left, upper, not transposed
  number *a, *b, *c;
  // The result is the same as lltx
  load_file("tests/xtrmm_ones_lltx.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('L', 'U', 'N', 'N', 32, 32, (number)1, a, 32, b, 32, flags);
  mu_assert("Error in test_xtrmm_ones_lunx(0).", equal_matrices(32, 32, b, 32, c, 32));

  load_file("tests/xtrmm_ones_lltx.txt", &a, &b, &c);

  // B == C (when diagonal of A is treated as ones)
  blas_xtrmm('L', 'U', 'N', 'U', 32, 32, (number)1, a, 32, b, 32, flags);
  mu_assert("Error in test_xtrmm_ones_lunx(1).", equal_matrices(32, 32, b, 32, c, 32));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_ones_lutx(int flags, number t) {
  // Left, upper, transposed
  number *a, *b, *c;
  // The result is the same as llnx
  load_file("tests/xtrmm_ones_llnx.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('L', 'U', 'T', 'N', 32, 32, (number)1, a, 32, b, 32, flags);
  mu_assert("Error in test_xtrmm_ones_lutx(0).", equal_matrices(32, 32, b, 32, c, 32));

  load_file("tests/xtrmm_ones_llnx.txt", &a, &b, &c);

  // B == C (when diagonal of A is treated as ones)
  blas_xtrmm('L', 'U', 'T', 'U', 32, 32, (number)1, a, 32, b, 32, flags);
  mu_assert("Error in test_xtrmm_ones_lutx(1).", equal_matrices(32, 32, b, 32, c, 32));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_ones_rlnx(int flags, number t) {
  // Right, lower, not transposed
  number *a, *b, *c;
  load_file("tests/xtrmm_ones_rlnx.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('R', 'L', 'N', 'N', 32, 32, (number)1, a, 32, b, 32, flags);
  mu_assert("Error in test_xtrmm_ones_rlnx(0).", equal_matrices(32, 32, b, 32, c, 32));

  load_file("tests/xtrmm_ones_rlnx.txt", &a, &b, &c);

  // B == C (when diagonal of A is treated as ones)
  blas_xtrmm('R', 'L', 'N', 'U', 32, 32, (number)1, a, 32, b, 32, flags);
  mu_assert("Error in test_xtrmm_ones_rlnx(1).", equal_matrices(32, 32, b, 32, c, 32));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_ones_rltx(int flags, number t) {
  // Right, lower, not transposed
  number *a, *b, *c;
  load_file("tests/xtrmm_ones_rltx.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('R', 'L', 'T', 'N', 32, 32, (number)1, a, 32, b, 32, flags);
  mu_assert("Error in test_xtrmm_ones_rltx(0).", equal_matrices(32, 32, b, 32, c, 32));

  load_file("tests/xtrmm_ones_rltx.txt", &a, &b, &c);

  // B == C (when diagonal of A is treated as ones)
  blas_xtrmm('R', 'L', 'T', 'U', 32, 32, (number)1, a, 32, b, 32, flags);
  mu_assert("Error in test_xtrmm_ones_rltx(1).", equal_matrices(32, 32, b, 32, c, 32));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_ones_runx(int flags, number t) {
  // Right, upper, non transposed
  number *a, *b, *c;
  // The result is the same as rltx
  load_file("tests/xtrmm_ones_rltx.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('R', 'U', 'N', 'N', 32, 32, (number)1, a, 32, b, 32, flags);
  mu_assert("Error in test_xtrmm_ones_runx(0).", equal_matrices(32, 32, b, 32, c, 32));

  load_file("tests/xtrmm_ones_rltx.txt", &a, &b, &c);

  // B == C (when diagonal of A is treated as ones)
  blas_xtrmm('R', 'U', 'N', 'U', 32, 32, (number)1, a, 32, b, 32, flags);
  mu_assert("Error in test_xtrmm_ones_runx(1).", equal_matrices(32, 32, b, 32, c, 32));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_ones_rutx(int flags, number t) {
  // Right, upper, transposed
  number *a, *b, *c;
  // The result is the same as rlnx
  load_file("tests/xtrmm_ones_rlnx.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('R', 'U', 'T', 'N', 32, 32, (number)1, a, 32, b, 32, flags);
  mu_assert("Error in test_xtrmm_ones_rutx(0).", equal_matrices(32, 32, b, 32, c, 32));

  load_file("tests/xtrmm_ones_rlnx.txt", &a, &b, &c);

  // B == C (when diagonal of A is treated as ones)
  blas_xtrmm('R', 'U', 'T', 'U', 32, 32, (number)1, a, 32, b, 32, flags);
  mu_assert("Error in test_xtrmm_ones_rutx(1).", equal_matrices(32, 32, b, 32, c, 32));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_rand_llnn(int flags, number t) {
  // C = 23x15
  number *a, *b, *c;
  load_file("tests/xtrmm_rand_llnn.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('L', 'L', 'N', 'N', 23, 15, (number)1, a, 23, b, 15, flags);
  mu_assert("Error in test_xtrmm_rand_llnn(0).", equal_matrices(23, 15, b, 15, c, 15));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_rand_llnu(int flags, number t) {
  // C = 13x7
  number *a, *b, *c;
  load_file("tests/xtrmm_rand_llnu.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('L', 'L', 'N', 'U', 13, 7, (number)1, a, 13, b, 7, flags);
  mu_assert("Error in test_xtrmm_rand_llnu(0).", equal_matrices(13, 7, b, 7, c, 7));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_rand_lltn(int flags, number t) {
  // C = 24x31
  number *a, *b, *c;
  load_file("tests/xtrmm_rand_lltn.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('L', 'L', 'T', 'N', 24, 31, (number)1, a, 24, b, 31, flags);
  mu_assert("Error in test_xtrmm_rand_lltn(0).", equal_matrices(24, 31, b, 31, c, 31));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_rand_lltu(int flags, number t) {
  // C = 40x20
  number *a, *b, *c;
  load_file("tests/xtrmm_rand_lltu.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('L', 'L', 'T', 'U', 40, 20, (number)1, a, 40, b, 20, flags);
  mu_assert("Error in test_xtrmm_rand_lltu(0).", equal_matrices(40, 20, b, 20, c, 20));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_rand_lunn(int flags, number t) {
  // C = 7x9
  number *a, *b, *c;
  load_file("tests/xtrmm_rand_lunn.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('L', 'U', 'N', 'N', 7, 9, (number)1, a, 7, b, 9, flags);
  mu_assert("Error in test_xtrmm_rand_lunn(0).", equal_matrices(7, 9, b, 9, c, 9));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_rand_lunu(int flags, number t) {
  // C = 35x33
  number *a, *b, *c;
  load_file("tests/xtrmm_rand_lunu.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('L', 'U', 'N', 'U', 35, 33, (number)1, a, 35, b, 33, flags);
  mu_assert("Error in test_xtrmm_rand_lunu(0).", equal_matrices(35, 33, b, 33, c, 33));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_rand_lutn(int flags, number t) {
  // C = 37x40
  number *a, *b, *c;
  load_file("tests/xtrmm_rand_lutn.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('L', 'U', 'T', 'N', 37, 40, (number)1, a, 37, b, 40, flags);
  mu_assert("Error in test_xtrmm_rand_lutn(0).", equal_matrices(37, 40, b, 40, c, 40));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_rand_lutu(int flags, number t) {
  // C = 37x40
  number *a, *b, *c;
  load_file("tests/xtrmm_rand_lutu.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('L', 'U', 'T', 'U', 37, 40, (number)1, a, 37, b, 40, flags);
  mu_assert("Error in test_xtrmm_rand_lutu(0).", equal_matrices(37, 40, b, 40, c, 40));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_rand_rlnn(int flags, number t) {
  // C = 33x39
  number *a, *b, *c;
  load_file("tests/xtrmm_rand_rlnn.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('R', 'L', 'N', 'N', 33, 39, (number)1, a, 39, b, 39, flags);
  mu_assert("Error in test_xtrmm_rand_rlnn(0).", equal_matrices(33, 39, b, 39, c, 39));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_rand_rlnu(int flags, number t) {
  // C = 34x38
  number *a, *b, *c;
  load_file("tests/xtrmm_rand_rlnu.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('R', 'L', 'N', 'U', 34, 38, (number)1, a, 38, b, 38, flags);
  mu_assert("Error in test_xtrmm_rand_rlnu(0).", equal_matrices(34, 38, b, 38, c, 38));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_rand_rltn(int flags, number t) {
  // C = 33x39
  number *a, *b, *c;
  load_file("tests/xtrmm_rand_rltn.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('R', 'L', 'T', 'N', 33, 39, (number)1, a, 39, b, 39, flags);
  mu_assert("Error in test_xtrmm_rand_rltn(0).", equal_matrices(33, 39, b, 39, c, 39));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_rand_rltu(int flags, number t) {
  // C = 34x38
  number *a, *b, *c;
  load_file("tests/xtrmm_rand_rltu.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('R', 'L', 'T', 'U', 34, 38, (number)1, a, 38, b, 38, flags);
  mu_assert("Error in test_xtrmm_rand_rltu(0).", equal_matrices(34, 38, b, 38, c, 38));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_rand_runn(int flags, number t) {
  // C = 57x59
  number *a, *b, *c;
  load_file("tests/xtrmm_rand_runn.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('R', 'U', 'N', 'N', 57, 59, (number)1, a, 59, b, 59, flags);
  mu_assert("Error in test_xtrmm_rand_runn(0).", equal_matrices(57, 59, b, 59, c, 59));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_rand_runu(int flags, number t) {
  // C = 57x59
  number *a, *b, *c;
  load_file("tests/xtrmm_rand_runu.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('R', 'U', 'N', 'U', 57, 59, (number)1, a, 59, b, 59, flags);
  mu_assert("Error in test_xtrmm_rand_runu(0).", equal_matrices(57, 59, b, 59, c, 59));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_rand_rutn(int flags, number t) {
  // C = 56x53
  number *a, *b, *c;
  load_file("tests/xtrmm_rand_rutn.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('R', 'U', 'T', 'N', 56, 53, (number)1, a, 53, b, 53, flags);
  mu_assert("Error in test_xtrmm_rand_rutn(0).", equal_matrices(56, 53, b, 53, c, 53));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_rand_rutu(int flags, number t) {
  // C = 56x53
  number *a, *b, *c;
  load_file("tests/xtrmm_rand_rutu.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('R', 'U', 'T', 'U', 56, 53, (number)1, a, 53, b, 53, flags);
  mu_assert("Error in test_xtrmm_rand_rutu(0).", equal_matrices(56, 53, b, 53, c, 53));

  free(a); free(b); free(c);
  return 0;
}

template <typename number>
static const char* test_xtrmm_left_row(int flags, number t) {
  // C = 1x129
  number *a, *b, *c;
  load_file("tests/xtrmm_left_row.txt", &a, &b, &c);

  // B == C
  blas_xtrmm('L', 'U', 'N', 'N', 1, 129, (number)1, a, 1, b, 129, flags);
  mu_assert("Error in test_xtrmm_left_row(0).", equal_matrices(1, 129, b, 129, c, 129));

  load_file("tests/xtrmm_left_row.txt", &a, &b, &c);
  // B == C (Upper A == Lower A)
  blas_xtrmm('L', 'L', 'N', 'N', 1, 129, (number)1, a, 1, b, 129, flags);
  mu_assert("Error in test_xtrmm_left_row(1).", equal_matrices(1, 129, b, 129, c, 129));

  load_file("tests/xtrmm_left_row.txt", &a, &b, &c);
  // B == C (A' == A)
  blas_xtrmm('L', 'L', 'T', 'N', 1, 129, (number)1, a, 1, b, 129, flags);
  mu_assert("Error in test_xtrmm_left_row(2).", equal_matrices(1, 129, b, 129, c, 129));

  load_file("tests/xtrmm_left_row.txt", &a, &b, &c);
  // B == C (Diag == 'U' and alpha = 34.7543)
  blas_xtrmm('L', 'U', 'T', 'U', 1, 129, (number)34.7543, a, 1, b, 129, flags);
  mu_assert("Error in test_xtrmm_left_row(3).", equal_matrices(1, 129, b, 129, c, 129));

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
    mu_run_test(test_xtrmm_ones_lunx, flags[i], t);
    mu_run_test(test_xtrmm_ones_lutx, flags[i], t);
    mu_run_test(test_xtrmm_ones_rlnx, flags[i], t);
    mu_run_test(test_xtrmm_ones_rltx, flags[i], t);
    mu_run_test(test_xtrmm_ones_runx, flags[i], t);
    mu_run_test(test_xtrmm_ones_rutx, flags[i], t);

    mu_run_test(test_xtrmm_rand_llnn, flags[i], t);
    mu_run_test(test_xtrmm_rand_llnu, flags[i], t);
    mu_run_test(test_xtrmm_rand_lltn, flags[i], t);
    mu_run_test(test_xtrmm_rand_lltu, flags[i], t);
    mu_run_test(test_xtrmm_rand_lunn, flags[i], t);
    mu_run_test(test_xtrmm_rand_lunu, flags[i], t);
    mu_run_test(test_xtrmm_rand_lutn, flags[i], t);
    mu_run_test(test_xtrmm_rand_lutu, flags[i], t);

    mu_run_test(test_xtrmm_rand_rlnn, flags[i], t);
    mu_run_test(test_xtrmm_rand_rlnu, flags[i], t);
    mu_run_test(test_xtrmm_rand_rltn, flags[i], t);
    mu_run_test(test_xtrmm_rand_rltu, flags[i], t);
    mu_run_test(test_xtrmm_rand_runn, flags[i], t);
    mu_run_test(test_xtrmm_rand_runu, flags[i], t);
    mu_run_test(test_xtrmm_rand_rutn, flags[i], t);
    mu_run_test(test_xtrmm_rand_rutu, flags[i], t);

    mu_run_test(test_xtrmm_left_row, flags[i], t);
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
