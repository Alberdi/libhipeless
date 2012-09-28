__kernel void matmul(__global float *a, __global const float *b, __global const float *c, const uint N) {
	float R; 
	int k;
	int xid = get_global_id(0);
	int yid = get_global_id(1);
	if (xid<N)
		R=0.0;
		for(k=0;k<N;k++) 
			R+=b[xid*N+k]*c[k*N+yid];
		a[xid*N+yid]=R;
}
