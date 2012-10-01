//TODO remove for final version
//#pragma OPENCL EXTENSION cl_intel_printf : enable

// TODO Puede no ser del todo correcto
// Thread block size
#define BLOCK_SIZE 16

__kernel void matmul(__global float *C, __global const float *A, __global const float *B, const uint colsA, const uint colsB) {
  float Csub = 0;

  // Block index
  int bx = get_group_id(0);
  int by = get_group_id(1);

  // Thread index
	int tx = get_local_id(0);
	int ty = get_local_id(1);

  // Index of the first sub-matrix of A processed by the block
  int aBegin = colsA * BLOCK_SIZE * by;
 
  // Index of the last sub-matrix of A processed by the block
  int aEnd   = aBegin + colsA - 1;
 
  // Step size used to iterate through the sub-matrices of A
  int aStep  = BLOCK_SIZE;
 
  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;
 
  // Step size used to iterate through the sub-matrices of B
  int bStep  = BLOCK_SIZE * colsB;


  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Declaration of the local memory array As 
    // used to store the sub-matrix of A
    __local float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the local memory array Bs 
    // used to store the sub-matrix of B
    __local float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from global memory
    // to local memory; each thread loads
    // one element of each matrix
    As[ty][tx] = A[a + colsA * ty + tx];
    Bs[ty][tx] = B[b + colsB * ty + tx];

    // Synchronize to make sure the matrices 
    // are loaded
    barrier(CLK_LOCAL_MEM_FENCE);

//    printf("a=%d, b=%d, Bs[15][0]=%f (%d)\n", a, b, Bs[15][0], b+colsB*15);
    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
    for (int k = 0; k < BLOCK_SIZE; k++) {
      Csub += As[ty][k] * Bs[k][tx];
//      printf("Csub= %f = A*B = %f * % f (Bs[%d][%d])\n", Csub, As[ty][k], Bs[k][tx], k, tx);
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = colsB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + colsB * ty + tx] = Csub;
}
