// Thread block size
#define BLOCK_SIZE 16

__kernel void blas_sgemm(__global float *C, __global const float *A, __global const float *B, const uint rowsA, const uint colsA, const uint colsB,
                        float alpha, float beta) {
  uint rowsB = colsA;
  uint ra, ca, rb, cb;
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

  // Declaration of the local memory array As 
  // used to store the sub-matrix of A
  __local float As[BLOCK_SIZE][BLOCK_SIZE];
                           
  // Declaration of the local memory array Bs 
  // used to store the sub-matrix of B
  __local float Bs[BLOCK_SIZE][BLOCK_SIZE];

  for(int i=0; i<colsA; i+=BLOCK_SIZE) {
    // Load the matrices from global memory to local memory;
    // each thread loads one element of each matrix
    // Barriers are used for synchronization and to be sure we don't
    // overwrite an address that's going to be used
    ra = tx+BLOCK_SIZE*bx;
    ca = i+ty;
    rb = i+tx;
    cb = ty+BLOCK_SIZE*by;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ra >= rowsA || ca >= colsA)
      As[tx][ty] = 0;
    else
      As[tx][ty] = A[ra*colsA+ca];
    if(rb >= rowsB || cb >= colsB)
      Bs[tx][ty] = 0;
    else
      Bs[tx][ty] = B[rb*colsB+cb];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int k=0; k<BLOCK_SIZE; k++)
      Csub += As[tx][k] * Bs[k][ty];
  }

  if(tx+a < rowsA && ty+b < colsB) // In bounds
    C[(tx+a)*colsB+(ty+b)] = alpha*Csub + beta*C[(tx+a)*colsB+(ty+b)];

}

__kernel void matmul(__global float *C, __global const float *A, __global const float *B, const uint rowsA, const uint colsA, const uint colsB) {
  uint rowsB = colsA;
  uint ra, ca, rb, cb;
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

  // Declaration of the local memory array As 
  // used to store the sub-matrix of A
  __local float As[BLOCK_SIZE][BLOCK_SIZE];
                           
  // Declaration of the local memory array Bs 
  // used to store the sub-matrix of B
  __local float Bs[BLOCK_SIZE][BLOCK_SIZE];

  for(int i=0; i<colsA; i+=BLOCK_SIZE) {
    // Load the matrices from global memory to local memory;
    // each thread loads one element of each matrix
    // Barriers are used for synchronization and to be sure we don't
    // overwrite an address that's going to be used
    ra = tx+BLOCK_SIZE*bx;
    ca = i+ty;
    rb = i+tx;
    cb = ty+BLOCK_SIZE*by;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ra >= rowsA || ca >= colsA)
      As[tx][ty] = 0;
    else
      As[tx][ty] = A[ra*colsA+ca];
    if(rb >= rowsB || cb >= colsB)
      Bs[tx][ty] = 0;
    else
      Bs[tx][ty] = B[rb*colsB+cb];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int k=0; k<BLOCK_SIZE; k++)
      Csub += As[tx][k] * Bs[k][ty];
  }

  if(tx+a < rowsA && ty+b < colsB) // In bounds
    C[(tx+a)*colsB+(ty+b)] = Csub;
}
