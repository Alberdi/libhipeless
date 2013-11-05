// Thread block size
#define BLOCK_SIZE 16

__kernel void blas_strmm(int left, int upper, int nota, int unit, int row, int dim, int m, int n,
                         float alpha, __global const float *a, __global const float *b, __global float *c) {

  float Csub = 0;

  // Thread index
  int tx = get_local_id(0);
  int ty = get_local_id(1);

  // Target index
  int x = tx + BLOCK_SIZE * get_group_id(0);
  int y = ty + BLOCK_SIZE * get_group_id(1);

  int ax = x;
  int ay = y;
  int bx = x;
  int by = y;
 
  // Declaration of the local memory array As 
  // used to store the sub-matrix of a
  __local float As[BLOCK_SIZE][BLOCK_SIZE];
                           
  // Declaration of the local memory array Bs 
  // used to store the sub-matrix of b
  __local float Bs[BLOCK_SIZE][BLOCK_SIZE];
  
  // If it's an upper triangular matrix, we can skip the first blocks full of zeroes.
  int start = left && upper == nota ? (x/BLOCK_SIZE) * BLOCK_SIZE : 0;
  // On lower triangular matrices, we can skip the last blocks full of zeroes.
  int end = !left || upper == nota ? dim : dim - ((row-2-x+tx)/BLOCK_SIZE) * BLOCK_SIZE;

  for(int i=start; i<end; i+=BLOCK_SIZE) {
    if(left) {
      ay = i+ty;
      bx = i+tx;
    }
    else {
      ax = i+tx;
      by = i+ty;
    }
    // Load the matrices from global memory to local memory; each thread loads one element of each matrix.
    // Barriers are used to be sure we don't overwrite an address that is going to be used.
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ax >= row || ay >= dim || (upper == nota && ay < ax) || (upper != nota && ay > dim-row+ax))
      As[tx][ty] = 0;
    else
      if(unit && ax == ay)
        As[tx][ty] = 1;
      else
        As[tx][ty] = nota ? a[ax*dim+ay] : a[ay*row+ax];

    if(bx >= m || by >= n)
      Bs[tx][ty] = 0;
    else
      Bs[tx][ty] = b[bx*n+by];
    barrier(CLK_LOCAL_MEM_FENCE);

    if(left)
      for(int l=0; l<BLOCK_SIZE; l++)
        Csub += As[tx][l] * Bs[l][ty];
    else
      for(int l=0; l<BLOCK_SIZE; l++)
        Csub += Bs[tx][l] * As[l][ty];
  }

  if(y < n && x < (left ? row : m)) // In bounds
    c[x*n+y] = alpha*Csub;
}

#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void blas_dtrmm(int left, int upper, int nota, int unit, int row, int dim, int m, int n,
                         double alpha, __global const double *a, __global const double *b, __global double *c) {

  double Csub = 0;

  // Thread index
  int tx = get_local_id(0);
  int ty = get_local_id(1);

  // Target index
  int x = tx + BLOCK_SIZE * get_group_id(0);
  int y = ty + BLOCK_SIZE * get_group_id(1);
 
  // Declaration of the local memory array As 
  // used to store the sub-matrix of a
  __local double As[BLOCK_SIZE][BLOCK_SIZE];
                           
  // Declaration of the local memory array Bs 
  // used to store the sub-matrix of b
  __local double Bs[BLOCK_SIZE][BLOCK_SIZE];

  // If it's an upper triangular matrix, we can skip the first blocks full of zeroes.
  int start = upper == nota ? (x/BLOCK_SIZE) * BLOCK_SIZE : 0;
  // On lower triangular matrices, we can skip the last blocks full of zeroes.
  int end = upper == nota ? dim : dim - ((row-1-x)/BLOCK_SIZE) * BLOCK_SIZE;

  for(int i=start; i<end; i+=BLOCK_SIZE) {
    // Load the matrices from global memory to local memory; each thread loads one element of each matrix.
    // Barriers are used to be sure we don't overwrite an address that is going to be used.
    barrier(CLK_LOCAL_MEM_FENCE);
    if(x >= row || i+ty >= dim || (upper == nota && i+ty < x) || (upper != nota && i+ty > dim-row+x))
      As[tx][ty] = 0;
    else {
      if(unit && x == i+ty)
        As[tx][ty] = 1;
      else
        As[tx][ty] = nota ? a[x*dim+i+ty] : a[(i+ty)*row+x];
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

