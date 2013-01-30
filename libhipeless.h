#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define BLOCK_SIZE 16

#define USE_CPU 0x01
#define USE_GPU 0x02
#define USE_MPI 0x04
#define NON_MPI_ROOT 0x08

typedef struct {
  size_t size1;
  size_t size2;
  cl_float *data;
} float_matrix;

// C = alpha * A * B + beta * C
// Single precission/float
void blas_sgemm(cl_char transa, cl_char transb, cl_int m, cl_int  n,  cl_int  k,
                cl_float alpha, cl_float *a, cl_int lda, cl_float *b, cl_int ldb,
                cl_float beta, cl_float *c, cl_int ldc, unsigned int flags);
