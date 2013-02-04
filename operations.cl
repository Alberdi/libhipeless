// Thread block size
#define BLOCK_SIZE 16

__kernel void blas_sgemm(int nota, int notb, int m, int n, int k, float alpha, __global const float *a, int lda,
                        __global const float *b, int ldb, float beta, __global float *c, int ldc) {

  uint ra, ca, rb, cb;
  float Csub = 0;

  // Block index
  int bx = get_group_id(0);
  int by = get_group_id(1);

  // Thread index
  int tx = get_local_id(0);
  int ty = get_local_id(1);

  // Index of the first sub-matrix of a processed by the block
  int indexa = BLOCK_SIZE * bx;
 
  // Index of the first sub-matrix of b processed by the block
  int indexb = BLOCK_SIZE * by;

  // Declaration of the local memory array As 
  // used to store the sub-matrix of a
  __local float As[BLOCK_SIZE][BLOCK_SIZE];
                           
  // Declaration of the local memory array Bs 
  // used to store the sub-matrix of b
  __local float Bs[BLOCK_SIZE][BLOCK_SIZE];

  for(int i=0; i<k; i+=BLOCK_SIZE) {
    // Load the matrices from global memory to local memory;
    // each thread loads one element of each matrix
    // Barriers are used for synchronization and to be sure we don't
    // overwrite an address that is going to be used
    ra = tx+BLOCK_SIZE*bx;
    ca = i+ty;
    rb = i+tx;
    cb = ty+BLOCK_SIZE*by;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(nota) {
      if(ra >= m || ca >= k)
        As[tx][ty] = 0;
      else
        As[tx][ty] = a[ra*k+ca];
    }
    else {
      if(ra >= k || ca >= m)
        As[tx][ty] = 0;
      else
        As[tx][ty] = a[ca*m+ra];
    }
    if(notb) {
      if(rb >= k || cb >= n)
        Bs[tx][ty] = 0;
      else
        Bs[tx][ty] = b[rb*n+cb];
    }
    else {
      if(cb >= k || rb >= n)
        Bs[tx][ty] = 0;
      else
        Bs[tx][ty] = b[cb*k+rb];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int l=0; l<BLOCK_SIZE; l++)
      Csub += As[tx][l] * Bs[l][ty];
  }
  if(tx+indexa < m && ty+indexb < n) { // In bounds
    if(beta)
      c[(tx+indexa)*n+(ty+indexb)] = alpha*Csub + beta*c[(tx+indexa)*n+(ty+indexb)];
    else
      c[(tx+indexa)*n+(ty+indexb)] = alpha*Csub;
  }
}

