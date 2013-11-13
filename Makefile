FLAGS = -O3 -l OpenCL

all: main mpihelper tests

main: main.c libhipeless.h
	mpic++ -o main main.c libhipeless.h $(FLAGS)

mpihelper: mpihelper.cpp libhipeless.h
	mpic++ -o mpihelper mpihelper.cpp libhipeless.h $(FLAGS)

test: test.cpp
	g++ -o test test.cpp $(FLAGS)

tests: tests.c
	mpic++ -o tests tests.c libhipeless.h $(FLAGS)

clean:
	rm main mpihelper test

