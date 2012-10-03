//TODO remove for final version
#pragma OPENCL EXTENSION cl_intel_printf : enable

// TODO Puede no ser del todo correcto
// Thread block size
#define BLOCK_SIZE 16

__kernel void matmul(__global float *C, __global const float *A, __global const float *B, const uint rowsA, const uint colsA, const uint colsB) {
  float Csub = 0;

  // Block index
  int bx = get_group_id(0);
  int by = get_group_id(1);

  // Thread index
  int tx = get_local_id(0);
  int ty = get_local_id(1);

  // Index of the first sub-matrix of A processed by the block
  int a = BLOCK_SIZE * bx;
 
  // Index of the first sub-matrix of B processed by the block
  int b = BLOCK_SIZE * by;

  for(int k=0; k<colsA; k++) {
    Csub += A[(tx+a)*colsA+k]*B[k*colsB+(ty+b)];
  }
 
  C[(tx+a)*colsB+(ty+b)] = Csub;
}
