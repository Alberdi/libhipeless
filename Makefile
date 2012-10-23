#FLAGS = -O3 -L/usr/lib64/OpenCL/vendors/intel/ -lOpenCL -I/usr/local/cuda/include/
FLAGS = -O3 -L /opt/AMDAPP/lib/x86_64/ -l OpenCL -I /opt/AMDAPP/include/

main: main.o libhipeless.o
	mpic++ -o main main.o libhipeless.o $(FLAGS)

main.o: main.c libhipeless.h
	mpic++ -c main.c $(FLAGS)

libhipeless.o: libhipeless.cpp libhipeless.h
	mpic++ -c libhipeless.cpp $(FLAGS)

clean:
	rm *.o main

