#include <stdio.h>

void helloCPU()
{
  printf("Hello from the CPU.\n");
}

/*
 * Refactor the `helloGPU` definition to be a kernel
 * that can be launched on the GPU. Update its message
 * to read "Hello from the GPU!"
 */

__global__ void helloGPU()
{
  printf("Hello from the GPU!\n");
}

int main()
{
    // regular flow
    //helloCPU();
    //helloGPU<<<1, 1>>>(); // <<< NUMBER_OF_BLOCKS, NUMBER_OF_THREADS_PER_BLOCK>>>
    //cudaDeviceSynchronize(); // host waits for kernel (gpu functions) execution to complete, else gpu never prints and cpu ends process
    
    // GPU prints before CPU
    //helloGPU<<<1, 1>>>(); 
    //cudaDeviceSynchronize();
    //helloCPU();
    
    // GPU prints before and after CPU
    helloGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
    helloCPU();
    helloGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
    

  /*
   * Refactor this call to `helloGPU` so that it launches
   * as a kernel on the GPU.
   */

  /*
   * Add code below to synchronize on the completion of the
   * `helloGPU` kernel completion before continuing the CPU
   * thread.
   */
   
   // run with
   // !nvcc -arch=sm_70 -o hello-gpu 01-hello/01-hello-gpu.cu -run
   
}
