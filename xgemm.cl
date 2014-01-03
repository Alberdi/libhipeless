#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void function(int nota, int notb, int m, int n, int k, number alpha, __global const number *a,
                       __global const number *b, number beta, __global number *c) {

  uint ca, rb;
  number Csub = 0;

  // Thread index
  int tx = get_local_id(0);
  int ty = get_local_id(1);

  // Target index
  int x = tx + BLOCK_SIZE * get_group_id(0);
  int y = ty + BLOCK_SIZE * get_group_id(1);
 
  // Declaration of the local memory array As 
  // used to store the sub-matrix of a
  __local number As[BLOCK_SIZE][BLOCK_SIZE];
                           
  // Declaration of the local memory array Bs 
  // used to store the sub-matrix of b
  __local number Bs[BLOCK_SIZE][BLOCK_SIZE];

  for(int i=0; i<k; i+=BLOCK_SIZE) {
    // Load the matrices from global memory to local memory; each thread loads one element of each matrix.
    // Barriers are used to be sure we don't overwrite an address that is going to be used.
    ca = i+ty;
    rb = i+tx;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(x >= m || ca >= k) // Out of bounds, treat as zero.
      As[tx][ty] = 0;
    else {
      if(nota)
        As[tx][ty] = a[x*k+ca];
      else
        As[tx][ty] = a[ca*m+x];
    }

    if(rb >= k || y >= n) // Out of bounds, treat as zero.
      Bs[tx][ty] = 0;
    else {
      if(notb)
        Bs[tx][ty] = b[rb*n+y];
      else
        Bs[tx][ty] = b[y*k+rb];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int l=0; l<BLOCK_SIZE; l++)
      Csub += As[tx][l] * Bs[l][ty];
  }
  if(x < m && y < n) { // In bounds
    if(beta)
      c[x*n+y] = alpha*Csub + beta*c[x*n+y];
    else
      c[x*n+y] = alpha*Csub;
  }
}

