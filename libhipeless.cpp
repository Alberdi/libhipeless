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

void mpi_spawn(MPI_Comm *intercomm, int *mpi_size) {
  char* universe_size = getenv("HIPELESS_UNIVERSE_SIZE");
  if(universe_size == NULL) {
    universe_size = getenv("MPI_UNIVERSE_SIZE");
    if(universe_size == NULL) {
      fprintf(stderr, "Neither HIPELESS_UNIVERSE_SIZE nor MPI_UNIVERSE_SIZE are set.\n");
      exit(EXIT_FAILURE);
    }
  }
  *mpi_size = atoi(universe_size);
  char* mpi_helper = (char *) "mpihelper";
  MPI_Comm_spawn(mpi_helper, MPI_ARGV_NULL, *mpi_size-1, MPI_INFO_NULL, 0,
                 MPI_COMM_SELF, intercomm, MPI_ERRCODES_IGNORE);
}

void opencl_intialize(cl_context *context, cl_uint *num_devices, size_t *size_devices, cl_device_id **devices, unsigned int flags) {
  cl_int errcode;
  cl_uint size_platforms;
  int i;
  
  errcode = clGetPlatformIDs(0, NULL, &size_platforms);
  cl_platform_id platforms[size_platforms];
  errcode |= clGetPlatformIDs(size_platforms, platforms, NULL);
  checkErr(errcode, "clGetPlatformIDs");

  cl_device_type chosen_type = flags & USE_CPU ? CL_DEVICE_TYPE_CPU :
                               flags & USE_GPU ? CL_DEVICE_TYPE_GPU :
                               flags & USE_ACCELERATOR ? CL_DEVICE_TYPE_ACCELERATOR :
                               flags & USE_DEFAULT_CL ? CL_DEVICE_TYPE_DEFAULT :
                               /* otherwhise */ CL_DEVICE_TYPE_ALL;

  for(i = 0; i < size_platforms; i++) {
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i], 0};
    *context = clCreateContextFromType(cps, chosen_type, NULL, NULL, &errcode);
    if(errcode == CL_SUCCESS) {
      // We found a suitable platform
      break;
    }
  }
  checkErr(errcode, "clCreateContextFromType");

  errcode = clGetContextInfo(*context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), num_devices, NULL);
  checkErr(errcode, "clGetContextInfo1");
  errcode = clGetContextInfo(*context, CL_CONTEXT_DEVICES, 0, NULL, size_devices);
  checkErr(errcode, "clGetContextInfo2");
  *devices = (cl_device_id *) malloc(*size_devices);
  errcode = clGetContextInfo(*context, CL_CONTEXT_DEVICES, *size_devices, *devices, NULL);
  checkErr(errcode, "clGetContextInfo3");
}

void opencl_finalize(cl_context context, cl_program program, cl_kernel kernel, cl_command_queue *command_queues, cl_uint num_devices,
                     cl_device_id *devices, cl_mem *memC) {
  int i;
  for(i=0; i < num_devices; i++) {
    clFinish(command_queues[i]);
    clReleaseCommandQueue(command_queues[i]);
  }

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseContext(context);

  free(command_queues);
  free(devices);
  free(memC);
}

void opencl_load_kernel(cl_context context, cl_program *program, cl_kernel *kernel, cl_device_id* devices, size_t size_devices,
                        const char *filename, const char *kernelfunction) {
  cl_int errcode;

  // Load the filen into source
  std::ifstream file(filename);
  checkErr(file.is_open() ? CL_SUCCESS : -1, "ifstream()");
  std::string sourceString(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
  const char *source = sourceString.c_str();

  size_t size_source[] = { strlen(source) };
  *program = clCreateProgramWithSource(context, 1, &source, size_source, &errcode);
  checkErr(errcode, "clCreateProgramWithSource");

  errcode = clBuildProgram(*program, size_devices/sizeof(cl_device_id), devices, NULL, NULL, NULL);
  if(errcode == CL_BUILD_PROGRAM_FAILURE) {
    // Determine the size of the log
    size_t log_size;
    clGetProgramBuildInfo(*program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    
    // Allocate memory for the log
    char *log = (char *) malloc(log_size);
    
    // Get the log
    clGetProgramBuildInfo(*program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    
    // Print the log
    printf("%s\n", log);
  }
  checkErr(errcode, "clBuildProgram");

  *kernel = clCreateKernel(*program, kernelfunction, &errcode);
  checkErr(errcode, "clCreateKernel");
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
  size_t local_work_size[2] = {BLOCK_SIZE, BLOCK_SIZE};

  // global_work_size[0] will be determined for each device on the platform,
  // because the last one might have a bit more of work to do.
  global_work_size[1] = n + (n % BLOCK_SIZE ? BLOCK_SIZE - (n % BLOCK_SIZE) : 0);

  opencl_intialize(&context, &num_devices, &size_devices, &devices, flags);
  opencl_load_kernel(context, &program, &kernel, devices, size_devices, "xgemm.cl", kernelfunction);
  
  dev_m = m/num_devices;
  last_dev_m = m - dev_m*(num_devices-1);

  memA = clCreateBuffer(context, CL_MEM_READ_ONLY, last_dev_m*k*sizeof(number), NULL, &errcode);
  checkErr(errcode, "clCreateBufferA");

  memB = clCreateBuffer(context, CL_MEM_READ_ONLY, k*n*sizeof(number), NULL, &errcode);
  checkErr(errcode, "clCreateBufferB");

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
      if(n == ldb) {
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

  opencl_finalize(context, program, kernel, command_queues, num_devices, devices, memC);
}

// C = alpha*op(A)*op(B) + beta*C
template <typename number>
void blas_xgemm(cl_char transa, cl_char transb, cl_int m, cl_int n, cl_int k,
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
    mpi_number = function == SGEMM ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Comm_get_parent(&parent);
    if(parent == MPI_COMM_NULL) {
      mpi_spawn(&intercomm, &mpi_size);
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
    if(parent == MPI_COMM_NULL) {
      MPI_Type_free(&transtype_a);
      MPI_Type_free(&transtype_b);
      MPI_Type_free(&transtype_c);
    }
    else {
      free(a);
      free(b);
      free(c);
    }
  }
}

// B = alpha*op(A)*B, or B = alpha*B*op(A)
template <typename number>
void dummy_xtrmm(cl_int m, cl_int n, number *a, number *b, cl_int ldb, cl_int rank) {
  int i, j;
  for(i = 0; i < m; i++) {
    for(j = 0; j < n; j++) {
      b[i*ldb + j] = i;
    }
  }
}

template <typename number>
void opencl_xtrmm(cl_int left, cl_int upper, cl_int nota, cl_int unit, cl_int row, cl_int dim, cl_int m,
                  cl_int n, number alpha, number *a, cl_int lda, number *b, cl_int ldb, unsigned int flags,
                  const char* kernelfunction) {
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

  int dev_row, last_dev_row, iter_row;
  int dev_row_a, dev_row_b, iter_row_a, iter_row_b;

  const char *source;
  size_t size_devices;
  size_t global_work_size[2];
  size_t local_work_size[2] = {BLOCK_SIZE, BLOCK_SIZE};

  // global_work_size[0] will be determined for each device on the platform,
  // because the last one might have a bit more of work to do.
  global_work_size[1] = n + (n % BLOCK_SIZE ? BLOCK_SIZE - (n % BLOCK_SIZE) : 0);

  opencl_intialize(&context, &num_devices, &size_devices, &devices, flags);
  opencl_load_kernel(context, &program, &kernel, devices, size_devices, "xtrmm.cl", kernelfunction);
  
  dev_row = (left ? row : m)/num_devices;
  last_dev_row = (left ? row : m) - dev_row*(num_devices-1);

  dev_row_a = left ? dev_row : 0;
  dev_row_b = left ? 0 : dev_row;

  memA = clCreateBuffer(context, CL_MEM_READ_ONLY, (left ? last_dev_row : dim)*dim*sizeof(number), NULL, &errcode);
  checkErr(errcode, "clCreateBufferA");

  memB = clCreateBuffer(context, CL_MEM_READ_ONLY, (left ? m : last_dev_row)*n*sizeof(number), NULL, &errcode);
  checkErr(errcode, "clCreateBufferB");
   
  command_queues = (cl_command_queue*) malloc(sizeof(cl_command_queue)*size_devices);
  memC = (cl_mem *) malloc(sizeof(cl_mem)*num_devices);
  for(i=0; i < num_devices; i++) {
    iter_row = i == num_devices-1 ? last_dev_row : dev_row;
    iter_row_a = left ? iter_row : dim;
    iter_row_b = left ? m : iter_row;
    global_work_size[0] = iter_row + (iter_row % BLOCK_SIZE ? BLOCK_SIZE - (iter_row % BLOCK_SIZE) : 0);

    command_queues[i] = clCreateCommandQueue(context, devices[i], CL_QUEUE_PROFILING_ENABLE, &errcode);
    checkErr(errcode, "clCreateCommandQueue");

    if(nota) {
      // Load full consecutive rows of a
      if(dim == lda) {
        // In this case, we can write it all in one call
        errcode = clEnqueueWriteBuffer(command_queues[i], memA, CL_TRUE, 0, iter_row_a*dim*sizeof(number), &a[i*dev_row_a*dim], 0, NULL, NULL);
      }
      else {
        for(l=0; l<iter_row_a; l++) {
          errcode = clEnqueueWriteBuffer(command_queues[i], memA, CL_TRUE, l*dim*sizeof(number), dim*sizeof(number), &a[(i*dev_row_a+l)*lda], 0, NULL, NULL);
        }
      }
    }
    else {
      // Load full consecutive columns of a
      for(l=0; l<dim; l++) {
        errcode = clEnqueueWriteBuffer(command_queues[i], memA, CL_TRUE, l*iter_row_a*sizeof(number), iter_row_a*sizeof(number), &a[l*lda+i*dev_row_a], 0, NULL, NULL);
      }
    }
    checkErr(errcode, "clEnqueueWriteBufferA");

    // Load full consecutive rows of b
    if(n == ldb) {
      // In this case, we can write it all in one call
      errcode = clEnqueueWriteBuffer(command_queues[i], memB, CL_TRUE, 0, iter_row_b*n*sizeof(number), b, 0, NULL, NULL);
    }
    else {
      for(l=0; l<iter_row_b; l++) {
        errcode = clEnqueueWriteBuffer(command_queues[i], memB, CL_TRUE, l*n*sizeof(number), n*sizeof(number), &b[(i*dev_row_b+l)*ldb], 0, NULL, NULL);
      }
    }
    checkErr(errcode, "clEnqueueWriteBufferB");

    // Temporal C memory is needed to not overwrite directly the B matrix
    memC[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, iter_row*n*sizeof(number), NULL, &errcode);
    checkErr(errcode, "clCreateBufferC");

    checkErr(clSetKernelArg(kernel, 0, sizeof(cl_int), &left), "clSetKernelArg0");
    checkErr(clSetKernelArg(kernel, 1, sizeof(cl_int), &upper), "clSetKernelArg1");
    checkErr(clSetKernelArg(kernel, 2, sizeof(cl_int), &nota), "clSetKernelArg2");
    checkErr(clSetKernelArg(kernel, 3, sizeof(cl_int), &unit), "clSetKernelArg3");
    checkErr(clSetKernelArg(kernel, 4, sizeof(cl_int), left ? &iter_row : &row), "clSetKernelArg4");
    checkErr(clSetKernelArg(kernel, 5, sizeof(cl_int), &dim), "clSetKernelArg5");
    checkErr(clSetKernelArg(kernel, 6, sizeof(cl_int), left ? &m : &iter_row), "clSetKernelArg6");
    checkErr(clSetKernelArg(kernel, 7, sizeof(cl_int), &n), "clSetKernelArg7");
    checkErr(clSetKernelArg(kernel, 8, sizeof(number), &alpha), "clSetKernelArg8");
    checkErr(clSetKernelArg(kernel, 9, sizeof(cl_mem), &memA), "clSetKernelArg9");
    checkErr(clSetKernelArg(kernel, 10, sizeof(cl_mem), &memB), "clSetKernelArg10");
    checkErr(clSetKernelArg(kernel, 11, sizeof(cl_mem), &memC[i]), "clSetKernelArg11");

    errcode = clEnqueueNDRangeKernel(command_queues[i], kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    checkErr(errcode, "clEnqueueNDRangeKernel");
  }

  for(i=0; i < num_devices; i++) {
    iter_row = i == num_devices-1 ? last_dev_row : dev_row;
    clFinish(command_queues[i]);
    // Store the calculated values of C in B
    if(n == ldb) {
      errcode = clEnqueueReadBuffer(command_queues[i], memC[i], CL_TRUE, 0, iter_row*n*sizeof(number), &b[i*dev_row*ldb], 0, NULL, NULL);
    }
    else {
      for(l=0; l<iter_row; l++) {
        errcode = clEnqueueReadBuffer(command_queues[i], memC[i], CL_TRUE, l*n*sizeof(number), n*sizeof(number), &b[(l+i*dev_row)*ldb], 0, NULL, NULL);
      }
    }
    checkErr(errcode, "clEnqueueReadBuffer");
  }

  opencl_finalize(context, program, kernel, command_queues, num_devices, devices, memC);
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
  MPI_Datatype mpi_number, transtype_a;

  function = sizeof(number) == sizeof(cl_float) ? STRMM : DTRMM;
  strcpy(operation, function == STRMM ? "blas_strmm" : "blas_dtrmm");

  left = side == 'L' || side == 'l';
  upper = uplo == 'U' || uplo == 'u';
  unit = diag == 'U' || diag == 'u' ? 1 : 0;
  nota = transa == 'N' || transa == 'n';

  dim = left ? m : n;
  row = m;

  if(flags & USE_MPI) {
    mpi_number = function == STRMM ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Comm_get_parent(&parent);
    if(parent == MPI_COMM_NULL) {
      mpi_spawn(&intercomm, &mpi_size);
      root_argument = MPI_ROOT;
      MPI_Bcast(&function, 1, MPI_INTEGER, root_argument, intercomm);
        
      rows = (int *) malloc(mpi_size*sizeof(int));

      if(left) {
        elems = (dim*dim+dim)/mpi_size;
        start = upper == nota ? 0 : mpi_size-1;
        end = upper == nota ? mpi_size-1 : 0;
        delta = upper == nota ? 1 : -1;
        for(i = start; i != end; i += delta) {
          // Calculate the consecutive rows to be processed by each processor.
          // The equation is derived and explained in the documentation.
          rows[i] = round((2*dim+1 - sqrt((2*dim+1)*(2*dim+1)-4*(elems)))/2);
          dim -= rows[i];
        }
        rows[end] = dim;
      }
      else {
        // When A is on the right, we need the full matrix.
        for(i=0; i < mpi_size; i++) {
          rows[i] = n;
        }
        spawns_m = m/mpi_size;
      }
    }
    else {
      intercomm = parent;
      root_argument = 0;
    }
  }

  if(flags & USE_MPI) {
    // Broadcast common parameters
    MPI_Bcast(left ? &m : &spawns_m, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&n, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&alpha, 1, mpi_number, root_argument, intercomm);
    MPI_Bcast(&flags, 1, MPI_UNSIGNED, root_argument, intercomm);
    MPI_Bcast(&left, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&upper, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&unit, 1, MPI_INTEGER, root_argument, intercomm);
    MPI_Bcast(&nota, 1, MPI_INTEGER, root_argument, intercomm);

    if(!nota) {
      // We'll need to use vectors to send the columns
      MPI_Type_vector(1, 1, lda, mpi_number, &transtype_a);
      MPI_Type_create_resized(transtype_a, 0, m*lda*sizeof(number), &transtype_a);
      MPI_Type_commit(&transtype_a);
    }

    if(parent == MPI_COMM_NULL) {
      row = 0;
      for(i = 0; i < mpi_size-1; i++) {
        row += left ? rows[i] : 0;
        dim = upper == nota ? (left ? m-row : n-row) : row+rows[i+1];
        MPI_Send(&rows[i+1], 1, MPI_INTEGER, i, XTRMM_TAG_DIM, intercomm);
        MPI_Send(&dim, 1, MPI_INTEGER, i, XTRMM_TAG_DIM, intercomm);
        // Send A, each node i needs rows[i] rows of A
        for(j = 0; j < rows[i+1]; j++) {
          if(nota) {
            if(upper) {
              MPI_Send(&a[(row+j)*lda + row + j + unit], dim-j-unit, mpi_number, i, XTRMM_TAG_DATA, intercomm);
            }
            else {
              MPI_Send(&a[(row+j)*lda], row+1+j-unit, mpi_number, i, XTRMM_TAG_DATA, intercomm);
            }
          }
          else { // TRANSA = 'T'
            MPI_Type_vector(upper ? row+1+j-unit : dim-j-unit, 1, lda, mpi_number, &transtype_a);
            MPI_Type_commit(&transtype_a);
            start = upper ? row+j : row+j+lda*(row+j);
            MPI_Send(&a[start], 1 , transtype_a, i, XTRMM_TAG_DATA, intercomm);
          }
        }
        if(left) {
          // Send B, we don't need to send the rows that would be multiplied by zero
          for(j = 0; j < dim; j++) {
            if(upper == nota) {
              // Only the last dim rows are needed
              MPI_Send(&b[(m-dim+j)*ldb], n, mpi_number, i, XTRMM_TAG_DATA, intercomm);
            }
            else {
              // Only the first dim rows are needed
              MPI_Send(&b[j*ldb], n, mpi_number, i, XTRMM_TAG_DATA, intercomm);
            }
          }
        }
        else { // If not left
          for(j = 0; j < spawns_m; j++) {
            MPI_Send(&b[(j+i*spawns_m+m)*ldb], n, mpi_number, i, XTRMM_TAG_DATA, intercomm);
          }
        }
      }
      // Restore values for parent operation
      dim = upper == nota ? (left ? m : n) : rows[0];
      row = rows[0];
      if(!left) {
        m = m - spawns_m*(mpi_size-1);
      }
    }
    else {
      MPI_Recv(&row, 1, MPI_INTEGER, 0, XTRMM_TAG_DIM, intercomm, MPI_STATUS_IGNORE);
      MPI_Recv(&dim, 1, MPI_INTEGER, 0, XTRMM_TAG_DIM, intercomm, MPI_STATUS_IGNORE);
      lda = nota ? dim : row;
      ldb = n;
      if(left) {
        m = dim;
      }
      else {
        m = spawns_m;
        n = dim;
      }
      a = (number *) malloc(row*dim*sizeof(number));
      b = (number *) malloc(m*n*sizeof(number));
      // Recv A
      for(j = 0; j < row; j++) {
        if(nota) {
          if(upper) {
            MPI_Recv(&a[j*lda + j + unit], dim-j-unit, mpi_number, 0, XTRMM_TAG_DATA, intercomm, MPI_STATUS_IGNORE);
          }
          else {
            MPI_Recv(&a[j*lda], dim-row+1+j-unit, mpi_number, 0, XTRMM_TAG_DATA, intercomm, MPI_STATUS_IGNORE);
          }
        }
        else { // TRANSA = 'T'
          MPI_Type_vector(upper ? dim-row+1+j-unit : dim-j-unit , 1, lda, mpi_number, &transtype_a);
          MPI_Type_commit(&transtype_a);
          start = upper ? j : j+lda*j;
          MPI_Recv(&a[start], 1, transtype_a, 0, XTRMM_TAG_DATA, intercomm, MPI_STATUS_IGNORE);
        }
      }
      // Recv B
      end = left ? dim : m;
      for(j = 0; j < end; j++) {
        MPI_Recv(&b[j*ldb], n, mpi_number, 0, XTRMM_TAG_DATA, intercomm, MPI_STATUS_IGNORE);
      }
    }
  }

  opencl_xtrmm(left, upper, nota, unit, row, dim, m, n, alpha, a, lda, b, ldb, flags, operation);

  if(flags & USE_MPI) {
    if(parent == MPI_COMM_NULL) {
      row = 0;
      // Recv B
      // We recover the chunks in order because otherwise we wouldn't know where to place them
      for(i = 0; i < mpi_size-1; i++) {
        if(left) {
          row += rows[i];
          end = rows[i+1];
        }
        else {
          // If i == 0 more rows could have been processed by the master
          if(i == 0) {
            row = m;
            end = spawns_m;
          }
          else {
            row += spawns_m; 
          }
        }
        for(j = 0; j < end; j++) {
          MPI_Recv(&b[(row+j)*ldb], n, mpi_number, i, XTRMM_TAG_DATA, intercomm, MPI_STATUS_IGNORE);
        }
      }
    }
    else {
      // Send B
      for(j = 0; j < m; j++) {
        MPI_Send(&b[j*ldb], n, mpi_number, 0, XTRMM_TAG_DATA, intercomm);
      }
      free(a);
      free(b);
    }
  }
}
