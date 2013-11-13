FLAGS = -O3 -l OpenCL

all: main mpihelper tester

main: main.c libhipeless.h
	mpic++ -o main main.c libhipeless.h $(FLAGS)

mpihelper: mpihelper.cpp libhipeless.h
	mpic++ -o mpihelper mpihelper.cpp libhipeless.h $(FLAGS)

test: test.cpp
	g++ -o test test.cpp $(FLAGS)

tester: tester.c
	mpic++ -o tester tester.c libhipeless.h $(FLAGS)

clean:
	rm main mpihelper test tester

