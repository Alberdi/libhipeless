#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#define XTRMM_TAG_DIM  3122
#define XTRMM_TAG_DATA 3123

inline void checkErr(cl_int errcode, const char* name) {
  if(errcode != CL_SUCCESS) {
    std::cerr << "ERROR: " << name << " (" << errcode << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
}

const char* readKernelFromSource(const char* source) {
    std::ifstream file(source);
    checkErr(file.is_open() ? CL_SUCCESS : -1, "ifstream()");
    std::string sourceString( std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
    return sourceString.c_str();
}

template <typename number>
int opencl_operation(cl_int nota, cl_int notb, cl_int m, cl_int n, cl_int k, number alpha, number *a, cl_int lda,
                     number *b, cl_int ldb, number beta, number *c, cl_int ldc, unsigned int flags, const char* kernelfunction) {
  int i, l;
  cl_uint num_devices;
  cl_int errcode;
  cl_context context;
  cl_device_id *devices;
  cl_command_queue *command_queues;
  cl_mem memA, memB;
  cl_mem *memC;
  cl_program program;
  cl_kernel kernel;

  int dev_m, last_dev_m, iter_m;

  const char *source;
  size_t size_devices;
  size_t global_work_size[2];
  size_t local_work_size[2];

  //global_work_size[0] = rowsA + (rowsA % BLOCK_SIZE ? BLOCK_SIZE - (rowsA % BLOCK_SIZE) : 0);
  global_work_size[1] = n + (n % BLOCK_SIZE ? BLOCK_SIZE - (n % BLOCK_SIZE) : 0);
  local_work_size[0] = BLOCK_SIZE;
  local_work_size[1] = BLOCK_SIZE;

  cl_uint size_platforms;
  errcode = clGetPlatformIDs(0, NULL, &size_platforms);
  cl_platform_id* platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id)*size_platforms);
  errcode |= clGetPlatformIDs(size_platforms, platforms, NULL);
  checkErr(errcode, "clGetPlatformIDs");
  // TODO Following line is not applicable to all the possible setups
  cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[flags&USE_CPU ? 0 : 1], 0};

  context = clCreateContextFromType(cps, flags&USE_CPU ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU, NULL, NULL, &errcode);
  checkErr(errcode, "clCreateContextFromType");

  errcode = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);
  checkErr(errcode, "clGetContextInfo1");
  errcode = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size_devices);
  checkErr(errcode, "clGetContextInfo2");
  devices = (cl_device_id *) malloc(size_devices);
  errcode = clGetContextInfo(context, CL_CONTEXT_DEVICES, size_devices, devices, NULL);
  checkErr(errcode, "clGetContextInfo3");

  dev_m = m/num_devices;
  last_dev_m = m - dev_m*(num_devices-1);

  memA = clCreateBuffer(context, CL_MEM_READ_ONLY, last_dev_m*k*sizeof(number), NULL, &errcode);
  checkErr(errcode, "clCreateBufferA");

  memB = clCreateBuffer(context, CL_MEM_READ_ONLY, k*n*sizeof(number), NULL, &errcode);
  checkErr(errcode, "clCreateBufferB");

  source = readKernelFromSource("operations.cl");
  size_t size_source[] = { strlen(source) };
  program = clCreateProgramWithSource(context, 1, &source, size_source, &errcode);
  checkErr(errcode, "clCreateProgramWithSource");

  errcode = clBuildProgram(program, size_devices/sizeof(cl_device_id), devices, NULL, NULL, NULL);
  checkErr(errcode, "clBuildProgram");

  kernel = clCreateKernel(program, kernelfunction, &errcode);
  checkErr(errcode, "clCreateKernel");
   
  command_queues = (cl_command_queue*) malloc(sizeof(cl_command_queue)*size_devices);
  memC = (cl_mem *) malloc(sizeof(cl_mem)*num_devices);
  for(i=0; i < num_devices; i++) {
    iter_m = i == num_devices-1 ? last_dev_m : dev_m;
    global_work_size[0] = iter_m + (iter_m % BLOCK_SIZE ? BLOCK_SIZE - (iter_m % BLOCK_SIZE) : 0);

    command_queues[i] = clCreateCommandQueue(context, devices[i], CL_QUEUE_PROFILING_ENABLE, &errcode);
    checkErr(errcode, "clCreateCommandQueue");

    if(nota) {
      // Load full consecutive rows of a
      if(k == lda) {
        // In this case, we can write it all in one call
        errcode = clEnqueueWriteBuffer(command_queues[i], memA, CL_TRUE, 0, iter_m*k*sizeof(number), &a[i*dev_m*k], 0, NULL, NULL);
      }
      else {
        for(l=0; l<iter_m; l++) {
          errcode = clEnqueueWriteBuffer(command_queues[i], memA, CL_TRUE, l*k*sizeof(number), k*sizeof(number), &a[(i*dev_m+l)*lda], 0, NULL, NULL);
        }
      }
    }
    else {
      // Load full consecutive columns of a
      for(l=0; l<k; l++) {
        errcode = clEnqueueWriteBuffer(command_queues[i], memA, CL_TRUE, l*iter_m*sizeof(number), iter_m*sizeof(number), &a[l*lda+i*dev_m], 0, NULL, NULL);
      }
    }
    checkErr(errcode, "clEnqueueWriteBufferA");

    if(notb) {
      // Load full consecutive rows of b
      if(n == lda) {
        // In this case, we can write it all in one call
        errcode = clEnqueueWriteBuffer(command_queues[i], memB, CL_TRUE, 0, k*n*sizeof(number), b, 0, NULL, NULL);
      }
      else {
        for(l=0; l<k; l++) {
          errcode = clEnqueueWriteBuffer(command_queues[i], memB, CL_TRUE, l*n*sizeof(number), n*sizeof(number), &b[l*ldb], 0, NULL, NULL);
        }
      }
    }
    else {
      // Load full consecutive columns of b
      for(l=0; l<n; l++) {
        errcode = clEnqueueWriteBuffer(command_queues[i], memB, CL_TRUE, l*k*sizeof(number), k*sizeof(number), &b[l*ldb], 0, NULL, NULL);
      }
    }
    checkErr(errcode, "clEnqueueWriteBufferB");

    // Load full consecutive rows of c
    memC[i] = clCreateBuffer(context, beta ? CL_MEM_READ_WRITE : CL_MEM_WRITE_ONLY, iter_m*n*sizeof(number), NULL, &errcode);
    checkErr(errcode, "clCreateBufferC");

    if(beta) {
      if(n == ldc) {
        // In this case, we can write it all in one call
        errcode = clEnqueueWriteBuffer(command_queues[i], memC[i], CL_TRUE, 0, iter_m*n*sizeof(number), &c[i*dev_m*n], 0, NULL, NULL);
      }
      else {
        for(l=0; l<iter_m; l++) {
          errcode = clEnqueueWriteBuffer(command_queues[i], memC[i], CL_TRUE, l*n*sizeof(number), n*sizeof(number), &c[(l+i*dev_m)*ldc], 0, NULL, NULL);
        }
      }
      checkErr(errcode, "clEnqueueWriteBufferC");
    }
  
    checkErr(clSetKernelArg(kernel, 0, sizeof(cl_int), &nota), "clSetKernelArg0");
    checkErr(clSetKernelArg(kernel, 1, sizeof(cl_int), &notb), "clSetKernelArg1");
    checkErr(clSetKernelArg(kernel, 2, sizeof(cl_int), &iter_m), "clSetKernelArg2");
    checkErr(clSetKernelArg(kernel, 3, sizeof(cl_int), &n), "clSetKernelArg3");
    checkErr(clSetKernelArg(kernel, 4, sizeof(cl_int), &k), "clSetKernelArg4");
    checkErr(clSetKernelArg(kernel, 5, sizeof(number), &alpha), "clSetKernelArg5");
    checkErr(clSetKernelArg(kernel, 6, sizeof(cl_mem), &memA), "clSetKernelArg6");
    checkErr(clSetKernelArg(kernel, 7, sizeof(cl_mem), &memB), "clSetKernelArg8");
    checkErr(clSetKernelArg(kernel, 8, sizeof(number), &beta), "clSetKernelArg10");
    checkErr(clSetKernelArg(kernel, 9, sizeof(cl_mem), &memC[i]), "clSetKernelArg11");

    errcode = clEnqueueNDRangeKernel(command_queues[i], kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    checkErr(errcode, "clEnqueueNDRangeKernel");
  }

  for(i=0; i < num_devices; i++) {
    iter_m = i == num_devices-1 ? last_dev_m : dev_m;
    clFinish(command_queues[i]);
    if(n == ldc) {
      errcode = clEnqueueReadBuffer(command_queues[i], memC[i], CL_TRUE, 0, iter_m*n*sizeof(number), &c[i*dev_m*ldc], 0, NULL, NULL);
    }
    else {
      for(l=0; l<iter_m; l++) {
        errcode = clEnqueueReadBuffer(command_queues[i], memC[i], CL_TRUE, l*n*sizeof(number), n*sizeof(number), &c[(l+i*dev_m)*ldc], 0, NULL, NULL);
      }
    }
    checkErr(errcode, "clEnqueueReadBuffer");
  }

  for(i=0; i < num_devices; i++) {
    clFinish(command_queues[i]);
    clReleaseCommandQueue(command_queues[i]);
  }

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseContext(context);
}

// C = alpha*op(A)*op(B) + beta*C
template <typename number>
void blas_xgemm(cl_char transa, cl_char transb, cl_int m, cl_int  n,  cl_int  k,
                number alpha, number *a, cl_int lda, number *b, cl_int ldb,
                number beta, number *c, cl_int ldc, unsigned int flags) {

  int root_argument, mpi_size, spawns_m, nota, notb;
  char operation[OPERATION_SIZE];
  int function;
  MPI_Comm intercomm, parent;
  MPI_Datatype transtype_a, transtype_b, transtype_c, mpi_number;

  function = sizeof(number) == sizeof(cl_float) ? SGEMM : DGEMM;
  strcpy(operation, function == SGEMM ? "blas_sgemm" : "blas_dgemm");

  nota = transa == 'N' || transa == 'n';
  notb = transb == 'N' || transb == 'n';

  if(flags & USE_MPI) {
    char* universe_size = getenv("MPI_UNIVERSE_SIZE");
    if(universe_size == NULL) {
      printf("MPI_UNIVERSE_SIZE is not set\n");
      return;
    }
    mpi_number = function == SGEMM ? MPI_FLOAT : MPI_DOUBLE;
    mpi_size = atoi(universe_size);
    MPI_Comm_get_parent(&parent);
    if(parent == MPI_COMM_NULL) {
      char* mpi_helper = (char *) "mpihelper";
      MPI_Comm_spawn(mpi_helper, MPI_ARGV_NULL, mpi_size-1, MPI_INFO_NULL, 0,
                    MPI_COMM_SELF, &intercomm, MPI_ERRCODES_IGNORE);
      root_argument = MPI_ROOT;
      spawns_m = m/mpi_size;

      MPI_Bcast(&function, 1, MPI_INTEGER, root_argument, intercomm);

      // We need to resize the types so MPI can know the real size of the elements.
      if(nota) {
        MPI_Type_vector(spawns_m, k, lda, mpi_number, &transtype_a);
        MPI_Type_create_resized(transtype_a, 0, spawns_m*lda*sizeof(number), &transtype_a);
      }
      else {
        MPI_Type_vector(k, spawns_m, lda, mpi_number, &transtype_a);
        MPI_Type_create_resized(transtype_a, 0, spawns_m*sizeof(number), &transtype_a);
      }
      MPI_Type_commit(&transtype_a);

      if(notb) {
        MPI_Type_vector(k, n, ldb, mpi_number, &transtype_b);
        MPI_Type_create_resized(transtype_b, 0, k*ldb*sizeof(number), &transtype_b);
      }
      else {
        MPI_Type_vector(n, k, ldb, mpi_number, &transtype_b);
        MPI_Type_create_resized(transtype_b, 0, k*sizeof(number), &transtype_b);
      }
      MPI_Type_commit(&transtype_b);

      MPI_Type_vector(spawns_m, n, ldc, mpi_number, &transtype_c);
      MPI_Type_create_resized(transtype_c, 0, spawns_m*ldc*sizeof(number), &transtype_c);
      MPI_Type_commit(&transtype_c);

      m = m - spawns_m*(mpi_size-1);
    }
    else {
      intercomm = parent;
      root_argument = 0;
    }
  }

  if(flags & USE_MPI) {
    // Broadcast needed parameters
    MPI_Bcast(&spawns_m, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&n, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&k, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&alpha, 1, mpi_number, root_argument, intercomm);
    MPI_Bcast(&beta, 1, mpi_number, root_argument, intercomm);
    MPI_Bcast(&flags, 1, MPI_UNSIGNED, root_argument, intercomm);
    MPI_Bcast(&nota, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&notb, 1, MPI_INTEGER, root_argument, intercomm);

    if(parent != MPI_COMM_NULL) {
      flags |= NON_MPI_ROOT;
      m = spawns_m;
      lda = nota ? k : m;
      ldb = notb ? n : k;
      ldc = n;
      a = (number *) malloc(m*k*sizeof(number));
      b = (number *) malloc(k*n*sizeof(number));
      c = (number *) malloc(m*n*sizeof(number));
    }

    if(nota) {
      // Send & Recv A, each node needs spawns_m rows of A
      MPI_Scatter(&a[m*lda], 1, transtype_a, a, m*k, mpi_number, root_argument, intercomm);
    }
    else {
      // Send & Recv A, each node needs spawns_m columns of A
      MPI_Scatter(&a[m], 1, transtype_a, a, spawns_m*k, mpi_number, root_argument, intercomm);
    }

    // Send B in full to each node
    if(parent == MPI_COMM_NULL) {
      // We need to use the custom datatype to send
      MPI_Bcast(b, 1, transtype_b, root_argument, intercomm);
    }
    else {
      // But we can receive a k*n array of floats
      MPI_Bcast(b, k*n, mpi_number, root_argument, intercomm);
    }

    if(beta) {
      // We also need to send C if beta != 0
      MPI_Scatter(&c[m*ldc], 1, transtype_c, c, m*n, mpi_number, root_argument, intercomm);
    }
  }
  
  opencl_operation(nota, notb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, flags, operation);

  if(flags & USE_MPI) {
    // Recv & Send C
    MPI_Gather(c, m*n, mpi_number, &c[m*ldc], 1, transtype_c, root_argument, intercomm);
    if(parent != MPI_COMM_NULL) {
      free(a);
      free(b);
      free(c);
    }
  }
}

// B = alpha*op(A)*B, or B = alpha*B*op(A)
template <typename number>
void blas_xtrmm(cl_char side, cl_char uplo, cl_char transa, cl_char diag, cl_int m,
                cl_int n, number alpha, number *a, cl_int lda, number *b, cl_int ldb,
                unsigned int flags) {
  int root_argument, mpi_size, spawns_m, left, upper, unit, nota, dim, i, j, elems, row;
  int start, end, delta;
  int *rows;
  char operation[OPERATION_SIZE];
  int function;
  MPI_Comm intercomm, parent;
  MPI_Datatype mpi_number;

  function = sizeof(number) == sizeof(cl_float) ? STRMM : DTRMM;
  strcpy(operation, function == STRMM ? "blas_strmm" : "blas_dtrmm");

  left = side == 'L' || side == 'l';
  upper = uplo == 'U' || uplo == 'u';
  unit = diag == 'U' || diag == 'u';
  nota = transa == 'N' || transa == 'n';

  if(flags & USE_MPI) {
    char* universe_size = getenv("MPI_UNIVERSE_SIZE");
    if(universe_size == NULL) {
      printf("MPI_UNIVERSE_SIZE is not set\n");
      return;
    }
    mpi_number = function == STRMM ? MPI_FLOAT : MPI_DOUBLE;
    mpi_size = atoi(universe_size);
    MPI_Comm_get_parent(&parent);
    if(parent == MPI_COMM_NULL) {
      char* mpi_helper = (char *) "mpihelper";
      MPI_Comm_spawn(mpi_helper, MPI_ARGV_NULL, mpi_size-1, MPI_INFO_NULL, 0,
                    MPI_COMM_SELF, &intercomm, MPI_ERRCODES_IGNORE);
      root_argument = MPI_ROOT;
      MPI_Bcast(&function, 1, MPI_INTEGER, root_argument, intercomm);

      rows = (int *) malloc(mpi_size*sizeof(int));
      dim = left ? m : n;
      elems = (dim*dim+dim)/mpi_size;
      start = upper ? 0 : mpi_size-1;
      end = upper ? mpi_size-1 : 0;
      delta = upper ? 1 : -1;
      for(i = start; i != end; i += delta) {
        // Calculate the consecutive rows to be processed by each processor.
        // The equation is derived and explained in the documentation.
        rows[i] = round((2*dim+1 - sqrt((2*dim+1)*(2*dim+1)-4*(elems)))/2);
        dim -= rows[i];
      }
      rows[i] = dim;
    }
    else {
      intercomm = parent;
      root_argument = 0;
    }
  }


  if(flags & USE_MPI) {
    // Broadcast common parameters
    //MPI_Bcast(&m, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&n, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&alpha, 1, mpi_number, root_argument, intercomm);
    MPI_Bcast(&flags, 1, MPI_UNSIGNED, root_argument, intercomm);
    MPI_Bcast(&left, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&upper, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&unit, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&nota, 1, MPI_INTEGER, root_argument, intercomm);

    if(parent == MPI_COMM_NULL) {
      row = 0;
      for(i = 0; i < mpi_size-1; i++) {
        row += rows[i];
        dim = upper ? m-row : row+rows[i];
        MPI_Send(&rows[i+1], 1, MPI_INTEGER, i, XTRMM_TAG_DIM, intercomm);
        MPI_Send(&dim, 1, MPI_INTEGER, i, XTRMM_TAG_DIM, intercomm);
        // Send A, each node i needs rows[i] rows of A
        for(j = 0; j < rows[i+1]; j++) {
          // We transmit the whole diagonal even though it might not be used
          if(nota) {
            if(upper) {
              MPI_Send(&a[(row+j)*lda + row + j], dim-j, mpi_number, i, XTRMM_TAG_DATA, intercomm);
            }
            else {
              //MPI_Send(&a[(row+j)*lda], row+rows[i+1], mpi_number, i, XTRMM_TAG_DATA, intercomm);
              MPI_Send(&a[(row+j)*lda], dim-rows[i+1]-1+j, mpi_number, i, XTRMM_TAG_DATA, intercomm);
            }
          }
        }
        // Send B, we don't need to send the rows that would be multiplied by zero
        // Only the last dim rows are needed
        for(j = 0; j < dim; j++) {
          MPI_Send(&b[(m-dim+j)*ldb], n, mpi_number, i, XTRMM_TAG_DATA, intercomm);
        }
      }
    }
    else {
      flags |= NON_MPI_ROOT;
      MPI_Recv(&row, 1, MPI_INTEGER, 0, XTRMM_TAG_DIM, intercomm, MPI_STATUS_IGNORE);
      MPI_Recv(&dim, 1, MPI_INTEGER, 0, XTRMM_TAG_DIM, intercomm, MPI_STATUS_IGNORE);
      lda = dim;
      ldb = n;
      m = dim;
      a = (number *) malloc(row*dim*sizeof(number));
      b = (number *) malloc(dim*n*sizeof(number));
      // Recv A
      for(j = 0; j < row; j++) {
        if(nota) {
          if(upper) {
            MPI_Recv(&a[j*lda + j], dim-j, mpi_number, 0, XTRMM_TAG_DATA, intercomm, MPI_STATUS_IGNORE);
          }
          else {
            MPI_Recv(&a[j*lda], dim-row-1+j, mpi_number, 0, XTRMM_TAG_DATA, intercomm, MPI_STATUS_IGNORE);
          }
        }
      }
      // Recv B
      for(j = 0; j < dim; j++) {
        MPI_Recv(&b[j*ldb], n, mpi_number, 0, XTRMM_TAG_DATA, intercomm, MPI_STATUS_IGNORE);
      }
    }
  }
  
  //opencl_operation(nota, notb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, flags, operation);

  if(flags & USE_MPI) {
    if(parent == MPI_COMM_NULL) {
      row = 0;
      // Recv B
      // We recover the chunks in order because otherwise we wouldn't know where to place them
      for(i = 0; i < mpi_size-1; i++) {
        row += rows[i];
        dim = upper ? m-row : row+rows[i];
        for(j = 0; j < dim; j++) {
          MPI_Recv(&b[(m-dim+j)*ldb], n, mpi_number, i, XTRMM_TAG_DATA, intercomm, MPI_STATUS_IGNORE);
        }
      }
    }
    else {
      // Send B
      for(j = 0; j < dim; j++) {
        MPI_Send(&b[j*ldb], n, mpi_number, 0, XTRMM_TAG_DATA, intercomm);
      }
      free(a);
      free(b);
    }
  }
}

