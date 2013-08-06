#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define SGEMM 1
#define DGEMM 2
#define STRMM 3
#define DTRMM 4

#define BLOCK_SIZE 16
#define OPERATION_SIZE 11

#define USE_CPU 0x01
#define USE_GPU 0x02
#define USE_MPI 0x04

#include "libhipeless.cpp"

// C = alpha * A * B + beta * C
// Double precission
void blas_dgemm(cl_char transa, cl_char transb, cl_int m, cl_int  n,  cl_int  k,
                cl_double alpha, cl_double *a, cl_int lda, cl_double *b, cl_int ldb,
                cl_double beta, cl_double *c, cl_int ldc, unsigned int flags) {
  blas_xgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, flags);
}

// C = alpha * A * B + beta * C
// Single precission/float
void blas_sgemm(cl_char transa, cl_char transb, cl_int m, cl_int  n,  cl_int  k,
                cl_float alpha, cl_float *a, cl_int lda, cl_float *b, cl_int ldb,
                cl_float beta, cl_float *c, cl_int ldc, unsigned int flags) {
  blas_xgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, flags);
}

// B = alpha*op(A)*B, or B = alpha*B*op(A)
// Double precission
void blas_dtrmm(cl_char side, cl_char uplo, cl_char transa, cl_char diag, cl_int m, cl_int n,
                cl_double alpha, cl_double *a, cl_int lda, cl_double *b, cl_int ldb, unsigned int flags) {
  blas_xtrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb, flags);
}

// B = alpha*op(A)*B, or B = alpha*B*op(A)
// Single precission/float
void blas_strmm(cl_char side, cl_char uplo, cl_char transa, cl_char diag, cl_int m, cl_int n,
                cl_float alpha, cl_float *a, cl_int lda, cl_float *b, cl_int ldb, unsigned int flags) {
  blas_xtrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb, flags);
}
