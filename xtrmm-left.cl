// Thread block size
#define BLOCK_SIZE 16

__kernel void blas_strmm(int upper, int nota, int unit, int row, int dim, int m, int n, float alpha,
                         __global const float *a, __global const float *b, __global float *c) {

  int ay, bx;
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
  
  // If it's an upper triangular matrix, we can skip the first blocks full of zeroes.
  int start = upper == nota ? (x/BLOCK_SIZE) * BLOCK_SIZE : 0;
  // On lower triangular matrices, we can skip the last blocks full of zeroes.
  int end = upper == nota ? dim : dim - ((row-2-x+tx)/BLOCK_SIZE) * BLOCK_SIZE;

  for(int i=start; i<end; i+=BLOCK_SIZE) {
    ay = i+ty;
    bx = i+tx;

    // Load the matrices from global memory to local memory; each thread loads one element of each matrix.
    // Barriers are used to be sure we don't overwrite an address that is going to be used.
    barrier(CLK_LOCAL_MEM_FENCE);
    if(x >= row || ay >= dim || (upper == nota && ay < x) || (upper != nota && ay > dim-row+x))
      As[tx][ty] = 0;
    else
      if(unit && ((upper == nota && x == ay) || (upper != nota && ay == dim-row+x)))
        As[tx][ty] = 1;
      else
        As[tx][ty] = nota ? a[x*dim+ay] : a[ay*row+x];

    if(bx >= m || y >= n)
      Bs[tx][ty] = 0;
    else
      Bs[tx][ty] = b[bx*n+y];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int l=0; l<BLOCK_SIZE; l++)
      Csub += As[tx][l] * Bs[l][ty];
  }

  if(y < n && x < row) // In bounds
    c[x*n+y] = alpha*Csub;
}

#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void blas_dtrmm(int upper, int nota, int unit, int row, int dim, int m, int n, double alpha,
                         __global const double *a, __global const double *b, __global double *c) {
  int ay, bx;
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
  int end = upper || upper == nota ? dim : dim - ((row-2-x+tx)/BLOCK_SIZE) * BLOCK_SIZE;

  for(int i=start; i<end; i+=BLOCK_SIZE) {
    ay = i+ty;
    bx = i+tx;

    // Load the matrices from global memory to local memory; each thread loads one element of each matrix.
    // Barriers are used to be sure we don't overwrite an address that is going to be used.
    barrier(CLK_LOCAL_MEM_FENCE);
    if(x >= row || ay >= dim || (upper == nota && ay < x) || (upper != nota && ay > dim-row+x))
      As[tx][ty] = 0;
    else
      if(unit && ((upper == nota && x == ay) || (upper != nota && ay == dim-row+x)))
        As[tx][ty] = 1;
      else
        As[tx][ty] = nota ? a[x*dim+ay] : a[ay*row+x];

    if(bx >= m || y >= n)
      Bs[tx][ty] = 0;
    else
      Bs[tx][ty] = b[bx*n+y];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int l=0; l<BLOCK_SIZE; l++)
      Csub += As[tx][l] * Bs[l][ty];
  }

  if(y < n && x < row) // In bounds
    c[x*n+y] = alpha*Csub;
}

