#FLAGS = -O3 -L/usr/lib64/OpenCL/vendors/intel/ -lOpenCL -I/usr/local/cuda/include/
FLAGS = -O3 -L /opt/AMDAPP/lib/x86_64/ -l OpenCL -I /opt/AMDAPP/include/

all: libhipeless.o main mpihelper

main: main.o libhipeless.o
	mpic++ -o main main.o libhipeless.o $(FLAGS)

main.o: main.c libhipeless.h
	mpic++ -c main.c $(FLAGS)

mpihelper: mpihelper.o libhipeless.o
	mpic++ -o mpihelper mpihelper.o libhipeless.o $(FLAGS)

mpihelper.o: mpihelper.cpp libhipeless.h
	mpic++ -c mpihelper.cpp $(FLAGS)

libhipeless.o: libhipeless.cpp libhipeless.h
	mpic++ -c libhipeless.cpp $(FLAGS)

clean:
	rm *.o main mpihelper

