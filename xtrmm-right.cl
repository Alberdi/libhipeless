#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void function(int upper, int nota, int unit, int row, int dim, int m, int n, number alpha,
                       __global const number *a, __global const number *b, __global number *c) {

  int ax, by;
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
  
 // If it's an upper triangular matrix, we can skip the first blocks full of zeroes.
 int start = upper != nota ? (y/BLOCK_SIZE) * BLOCK_SIZE : 0;
 // On lower triangular matrices, we can skip the last blocks full of zeroes.
 int end = upper != nota ? dim : dim - ((dim-2-y+ty)/BLOCK_SIZE) * BLOCK_SIZE;

  for(int i=start; i<end; i+=BLOCK_SIZE) {
    ax = i+tx;
    by = i+ty;

    // Load the matrices from global memory to local memory; each thread loads one element of each matrix.
    // Barriers are used to be sure we don't overwrite an address that is going to be used.
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ax >= dim || y >= dim || (upper == nota && y < ax) || (upper != nota && y > ax))
      As[tx][ty] = 0;
    else
      if(unit && y == ax)
        As[tx][ty] = 1;
      else
        As[tx][ty] = nota ? a[ax*dim+y] : a[y*dim+ax];

    if(x >= m || by >= n)
      Bs[tx][ty] = 0;
    else
      Bs[tx][ty] = b[x*n+by];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int l=0; l<BLOCK_SIZE; l++)
      Csub += Bs[tx][l] * As[l][ty];
  }

  if(y < n && x < m) // In bounds
    c[x*n+y] = alpha*Csub;
}

