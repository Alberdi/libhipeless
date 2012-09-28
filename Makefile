#FLAGS= -O3 -L/opt/cesga/AMD-APP-SDK-v2.4-lnx64/lib/x86_64/ -lOpenCL -I/opt/cesga/AMD-APP-SDK-v2.4-lnx64/include/
#FLAGS = -O3 -L/usr/local/cuda/lib64 -lOpenCL -I/usr/local/cuda/include/
#FLAGS = -O3 -L/usr/lib64/OpenCL/vendors/intel/ -lOpenCL -I/usr/local/cuda/include/
FLAGS = -O3 -L /usr/lib64/OpenCL/vendors/intel/ -l OpenCL -I /usr/local/cuda/include/

vec_sum: matmul.cpp
	mpic++  $(FLAGS) -o matmul.o matmul.cpp

clean:
	rm matmul.o

