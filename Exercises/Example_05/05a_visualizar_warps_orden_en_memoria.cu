#include <iostream>
#include <cuda.h>

__global__ void kernel_guardar_orden(int* salida){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    salida[tid] = tid; // Cada hilo escribe su propio ID
}

int main() {

    const int N = 64; // Numero total de hilos

    int threads_per_block = 32;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    // Reservar memoria en CPU
    int* h_salida = new int[N];

    // Reservar memoria en GPU
    int* d_salida;
    cudaMalloc(&d_salida, N * sizeof(int));

    // Lanzar el kernel
    kernel_guardar_orden <<< blocks, threads_per_block >>> (d_salida);

    //Esperar a que termine el kernel
    cudaDeviceSynchronize();

    // Copiar resultados de GPU al CPU
    cudaMemcpy(h_salida, d_salida, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Imprimir resultados
    for (int i = 0; i < N; ++i){
        std::cout << h_salida[i] << " ";
    }
    std::cout<< std::endl;

    // Liberar memoria
    delete[] h_salida;
    cudaFree(d_salida);

    return 0;
}

// nvcc guardar_orden.cu -o guardar_orden
