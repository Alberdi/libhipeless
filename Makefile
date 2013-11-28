FLAGS = -O3 -l OpenCL

all: main mpihelper tester

main: main.c libhipeless.h
	mpic++ -o main main.c libhipeless.h $(FLAGS)

mpihelper: mpihelper.cpp libhipeless.h
	mpic++ -o mpihelper mpihelper.cpp libhipeless.h $(FLAGS)

tester: tester.cpp libhipeless.h
	mpic++ -o tester tester.cpp libhipeless.h $(FLAGS)

clean:
	rm main mpihelper tester

