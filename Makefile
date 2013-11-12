FLAGS = -O3 -l OpenCL

all: main mpihelper tests

main: main.c libhipeless.h
	mpic++ -o main main.c libhipeless.h $(FLAGS)

mpihelper: mpihelper.cpp libhipeless.h
	mpic++ -o mpihelper mpihelper.cpp libhipeless.h $(FLAGS)

test: test.cpp
	g++ -o test test.cpp $(FLAGS)

tests: tests_xgemm.c
	mpic++ -o tests_xgemm tests_xgemm.c libhipeless.h $(FLAGS)

clean:
	rm main mpihelper test

