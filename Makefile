#FLAGS= -O3 -L/opt/cesga/AMD-APP-SDK-v2.4-lnx64/lib/x86_64/ -lOpenCL -I/opt/cesga/AMD-APP-SDK-v2.4-lnx64/include/
#FLAGS = -O3 -L/usr/local/cuda/lib64 -lOpenCL -I/usr/local/cuda/include/
#FLAGS = -O3 -L/usr/lib64/OpenCL/vendors/intel/ -lOpenCL -I/usr/local/cuda/include/
FLAGS = -O3 -L /opt/AMDAPP/lib/x86_64/ -l OpenCL -I /opt/AMDAPP/include/

main: main.c
	mpic++ -o main.o main.c libhipeless.cpp $(FLAGS)

clean:
	rm main.o

