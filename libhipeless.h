#ifndef HIPELESS_SEEN
#define HIPELESS_SEEN

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

#define USE_CPU 0x01
#define USE_GPU 0x02
#define USE_ACCELERATOR 0x04
#define USE_DEFAULT_CL 0x08
#define USE_ANY_CL 0x010

#define USE_MPI 0x20
#define MPI_SPAWN 0x40

// Error codes
#define HIPELESS_SUCCESS 0
#define HIPELESS_INVALID_VALUE_M -1
#define HIPELESS_INVALID_VALUE_N -2
#define HIPELESS_INVALID_VALUE_K -3
#define HIPELESS_INVALID_VALUE_LDA -4
#define HIPELESS_INVALID_VALUE_LDB -5
#define HIPELESS_INVALID_VALUE_LDC -6

#include "libhipeless.cpp"

extern "C" {

// C = alpha * A * B + beta * C
// Double precission
int blas_dgemm(cl_char transa, cl_char transb, cl_int m, cl_int  n,  cl_int  k, cl_double alpha, cl_double *a,
                cl_int lda, cl_double *b, cl_int ldb, cl_double beta, cl_double *c, cl_int ldc, unsigned int flags) {
  return blas_xgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, flags);
}

// C = alpha * A * B + beta * C
// Single precission/float
int blas_sgemm(cl_char transa, cl_char transb, cl_int m, cl_int  n,  cl_int  k, cl_float alpha, cl_float *a,
                cl_int lda, cl_float *b, cl_int ldb, cl_float beta, cl_float *c, cl_int ldc, unsigned int flags) {
  return blas_xgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, flags);
}

// B = alpha*op(A)*B, or B = alpha*B*op(A)
// Double precission
int blas_dtrmm(cl_char side, cl_char uplo, cl_char transa, cl_char diag, cl_int m, cl_int n,
                cl_double alpha, cl_double *a, cl_int lda, cl_double *b, cl_int ldb, unsigned int flags) {
  return blas_xtrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb, flags);
}

// B = alpha*op(A)*B, or B = alpha*B*op(A)
// Single precission/float
int blas_strmm(cl_char side, cl_char uplo, cl_char transa, cl_char diag, cl_int m, cl_int n,
                cl_float alpha, cl_float *a, cl_int lda, cl_float *b, cl_int ldb, unsigned int flags) {
  return blas_xtrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb, flags);
}

} // extern "C"

#endif /* !HIPELESS_SEEN */
