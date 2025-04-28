#include <iostream>
#include <cuda.h>

__global__ void kernel_visualizar() {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	printf("Hola desde el hilo %d (threadIdx.x=%d, blockIdx.x=%d)\n", tid, threadIdx.x, blockIdx.x);
}

int main(){
	const int N = 64; // Numero total de hilos

	int threads_per_block = 32; // Probaremos cambiar este valor
	int blocks = (N + threads_per_block -1) / threads_per_block;

	kernel_visualizar <<< blocks, threads_per_block >>> ();
	cudaDeviceSynchronize(); // Esperar que terminen todos los hilos

	return 0;
}

// nvcc 05_visualizar_warps.cu -o visualizar_warps