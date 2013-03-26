// Thread block size
#define BLOCK_SIZE 16

__kernel void blas_sgemm(int nota, int notb, int m, int n, int k, float alpha, __global const float *a,
                         __global const float *b, float beta, __global float *c) {

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
    if(ra >= m || ca >= k)
      As[tx][ty] = 0;
    else {
      if(nota)
        As[tx][ty] = a[ra*k+ca];
      else
        As[tx][ty] = a[ca*m+ra];
    }

    if(rb >= k || cb >= n)
      Bs[tx][ty] = 0;
    else {
      if(notb)
        Bs[tx][ty] = b[rb*n+cb];
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

__kernel void blas_dgemm(int nota, int notb, int m, int n, int k, double alpha, __global const double *a,
                         __global const double *b, double beta, __global double *c) {

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
    if(ra >= m || ca >= k)
      As[tx][ty] = 0;
    else {
      if(nota)
        As[tx][ty] = a[ra*k+ca];
      else
        As[tx][ty] = a[ca*m+ra];
    }

    if(rb >= k || cb >= n)
      Bs[tx][ty] = 0;
    else {
      if(notb)
        Bs[tx][ty] = b[rb*n+cb];
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

__kernel void blas_strmm(int left, int upper, int nota, int unit, int row, int dim, int m, int n,
                         float alpha, __global const float *a, __global const float *b, __global float *c) {

  float Csub = 0;

  // Thread index
  int tx = get_local_id(0);
  int ty = get_local_id(1);

  // Target index
  int x = tx + BLOCK_SIZE * get_group_id(0);
  int y = ty + BLOCK_SIZE * get_group_id(1);
 
  // Declaration of the local memory array As 
  // used to store the sub-matrix of a
  __local float As[BLOCK_SIZE][BLOCK_SIZE];
                           
  // Declaration of the local memory array Bs 
  // used to store the sub-matrix of b
  __local float Bs[BLOCK_SIZE][BLOCK_SIZE];

  for(int i=0; i<dim; i+=BLOCK_SIZE) {
    // Load the matrices from global memory to local memory;
    // each thread loads one element of each matrix
    // Barriers are used for synchronization and to be sure we don't
    // overwrite an address that is going to be used
    barrier(CLK_LOCAL_MEM_FENCE);
    if(x >= row || i+ty >= dim || i+ty < x)
      As[tx][ty] = 0;
    else {
      if(unit && x == i+ty)
        As[tx][ty] = 1;
      else
        As[tx][ty] = a[x*dim+i+ty];
    }
    if(i+tx >= m || y >= n)
      Bs[tx][ty] = 0;
    else
      Bs[tx][ty] = b[(i+tx)*n+y];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int l=0; l<BLOCK_SIZE; l++)
      Csub += As[tx][l] * Bs[l][ty];
  }
  if(x < m && y < n) { // In bounds
    c[x*n+y] = alpha*Csub;
  }
}

__kernel void blas_dtrmm(int left, int upper, int nota, int unit, int row, int dim, int m, int n,
                         double alpha, __global const double *a, __global const double *b, __global double *c) {
}

