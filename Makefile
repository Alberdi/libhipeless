FLAGS = -O3 -L /opt/AMDAPP/lib/x86_64/ -l OpenCL -I /opt/AMDAPP/include/

all: main mpihelper

main: main.c libhipeless.h
	mpic++ -o main main.c libhipeless.h $(FLAGS)

mpihelper: mpihelper.cpp libhipeless.h
	mpic++ -o mpihelper mpihelper.cpp libhipeless.h $(FLAGS)

test: test.cpp
	g++ -o test test.cpp $(FLAGS)

clean:
	rm main mpihelper test

