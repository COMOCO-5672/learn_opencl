#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

const int ARRAY_SIZE = 1000;

cl_context CreateContext(cl_device_id *device) {
  cl_int err_num;
  cl_uint num_platforms;
  cl_platform_id frist_platform_id;
  cl_context context = NULL;

  err_num = clGetPlatformIDs(1, &frist_platform_id, &num_platforms);
  if (err_num != CL_SUCCESS || num_platforms <= 0) {
    printf("Failed to find any OPENCL platforms\n");
    return NULL;
  }
  err_num =
      clGetDeviceIDs(frist_platform_id, CL_DEVICE_TYPE_GPU, 1, device, NULL);
  if (err_num != CL_SUCCESS) {
    printf("aThis is no GPU, trying CPU\n");
    err_num =
        clGetDeviceIDs(frist_platform_id, CL_DEVICE_TYPE_CPU, 1, device, NULL);
    if (err_num != CL_SUCCESS) {
      printf("No OpenCL compatible CPU or GPU device available\n");
      return NULL;
    }
  }

  context = clCreateContext(NULL, 1, device, NULL, NULL, &err_num);
  if (err_num != CL_SUCCESS) {
    printf("Create context error\n");
    return NULL;
  }
  return context;
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id devices) {
  cl_int err_num;
  cl_command_queue commandQueue = NULL;

  commandQueue =
      clCreateCommandQueueWithProperties(context, devices, 0, &err_num);
  if (err_num != CL_SUCCESS || commandQueue == NULL) {
    printf("Faile to create commandQueue for device 0\n");
    return NULL;
  }
  return commandQueue;
}

char *ReadKernalSourceFile(const char *filename, size_t *length) {
  FILE *file = NULL;
  size_t source_len;
  char *source_str;
  int ret;
  file = fopen(filename, "rb");
  if (file == NULL) {
    printf("%s at %d : can`t open %s", __FILE__, __LINE__ - 2, filename);
    return NULL;
  }

  // 获取文件大小
  fseek(file, 0, SEEK_END);
  source_len = ftell(file);
  fseek(file, 0, SEEK_SET);

  // 分配内存并读取文件
  source_str = (char *)malloc(source_len + 1);
  if (fread(source_str, source_len, 1, file) != 1) {
    printf("%s at %d : Can't read source %s\n", __FILE__, __LINE__ - 2,
           filename);
    fclose(file);
    free(source_str);
    return NULL;
  }

  fclose(file);
  if (length != 0) {
    *length = source_len;
  }
  source_str[source_len] = '\0';
  return source_str;
}

cl_program CreateProgram(cl_context context, cl_device_id device,
                         const char *filename) {
  cl_int err_num;
  cl_program program;
  size_t program_len;
  char *const source = ReadKernalSourceFile(filename, &program_len);
  program =
      clCreateProgramWithSource(context, 1, (const char **)&source, NULL, NULL);

  if (program == NULL) {
    printf("Failed to create CL program from source.\n");
    return NULL;
  }

  err_num = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err_num != CL_SUCCESS) {
    char build_log[16384];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          sizeof(build_log), build_log, NULL);
    printf("Error in kernel : %s \n", build_log);
    clReleaseProgram(program);
    return NULL;
  }

  return program;
}

bool CreateMemObjects(cl_context context, cl_mem mem_obj[3], float *a,
                      float *b) {

  mem_obj[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(float) * ARRAY_SIZE, a, NULL);
  mem_obj[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(float) * ARRAY_SIZE, b, NULL);
  mem_obj[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                              sizeof(float) * ARRAY_SIZE, NULL, NULL);

  if (mem_obj[0] == NULL || mem_obj[1] == NULL || mem_obj[2] == NULL) {
    printf("Error create memory objects\n");
    return false;
  }
  return true;
}

void CleanUp(cl_context context, cl_command_queue command_queue,
             cl_program program, cl_kernel kernal, cl_mem mem_obj[3]) {
  for (int i = 0; i < 3; i++) {
    if (mem_obj[i] != 0) {
      clReleaseMemObject(mem_obj[i]);
    }
  }

  if (command_queue != 0) {
    clReleaseCommandQueue(command_queue);
  }

  if (kernal != 0) {
    clReleaseKernel(kernal);
  }

  if (program != 0) {
    clReleaseProgram(program);
  }

  if (context != 0) {
    clReleaseContext(context);
  }
}

int main() {
  cl_context context = 0;
  cl_command_queue command_queue = 0;
  cl_program program = 0;
  cl_device_id device = 0;
  cl_kernel kernel = 0;
  cl_mem mem_objs[3] = {0, 0, 0};
  cl_int err_num;

  float *a = (float *)malloc(sizeof(float) * ARRAY_SIZE);
  float *b = (float *)malloc(sizeof(float) * ARRAY_SIZE);
  float *result = (float *)malloc(sizeof(float) * ARRAY_SIZE);

  for (int i = 0; i < ARRAY_SIZE; i++) {
    a[i] = i;
    b[i] = i * 2;
  }

  context = CreateContext(&device);
  if (context == NULL) {
    printf("Failed to create OpenCL conetxt.\n");
    return 1;
  }

  command_queue = CreateCommandQueue(context, device);
  if (command_queue == NULL) {
    CleanUp(context, command_queue, program, kernel, mem_objs);
    return 1;
  }

  program = CreateProgram(context, device, "vecAdd.cl");
  if (program == NULL) {
    CleanUp(context, command_queue, program, kernel, mem_objs);
    return 1;
  }

  kernel = clCreateKernel(program, "vecAdd", NULL);
  if (kernel == NULL) {
    printf("Failed to create kernel\n");
    CleanUp(context, command_queue, program, kernel, mem_objs);
    return 1;
  }

  if (!CreateMemObjects(context, mem_objs, a, b)) {
    CleanUp(context, command_queue, program, kernel, mem_objs);
    return 1;
  }

  err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_objs[0]);
  err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_objs[1]);
  err_num |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_objs[2]);
  if (err_num != CL_SUCCESS) {
    printf("Error setting kernel arguments\n");
    CleanUp(context, command_queue, program, kernel, mem_objs);
    return 1;
  }

  size_t global_work_size[1] = {ARRAY_SIZE};
  err_num = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                                   global_work_size, NULL, 0, NULL, NULL);
  if (err_num != CL_SUCCESS) {
    printf("Error queuing kernel for execution\n");
    CleanUp(context, command_queue, program, kernel, mem_objs);
    return 1;
  }

  err_num =
      clEnqueueReadBuffer(command_queue, mem_objs[2], CL_TRUE, 0,
                          ARRAY_SIZE * sizeof(float), result, 0, NULL, NULL);
  if (err_num != CL_SUCCESS) {
    printf("Error reading result buffer\n");
    CleanUp(context, command_queue, program, kernel, mem_objs);
    return 1;
  }

  printf("Computation completed. Verifying...\n");
  for (int i = 0; i < ARRAY_SIZE; i++) {
    if (result[i] != a[i] + b[i]) {
      printf("Verification failed at index %d\n", i);
      break;
    }
  }

  printf("Verification completed successfully!\n");

  CleanUp(context, command_queue, program, kernel, mem_objs);
  free(a);
  free(b);
  free(result);

  return 0;
}