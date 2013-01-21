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
int blas_sgemm(void* TransA, void* TransB, cl_float alpha, float_matrix *A, float_matrix *B, cl_float beta, float_matrix *C, unsigned int flags);

