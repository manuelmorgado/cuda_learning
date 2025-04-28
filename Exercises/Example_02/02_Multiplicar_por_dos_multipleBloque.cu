#include <iostream>
#include <cuda_runtime.h>

__global__ void multiplicar_por_dos(int *datos, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        datos[idx] *= 2;
    }
}

int main() {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Máximo hilos por bloque: " << prop.maxThreadsPerBlock << std::endl;

    int N = 1000;
    int *h_data = new int[N];
    int *d_data;

    // Inicializar
    for (int i = 0; i < N; ++i)
        h_data[i] = i;

    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // Lanzar muchos bloques e hilos
    int threads_per_block = prop.maxThreadsPerBlock;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    multiplicar_por_dos<<<blocks, threads_per_block>>>(d_data, N);

    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Primeros N resultados:\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_data[i] << " ";
    std::cout << std::endl;

    cudaFree(d_data);
    delete[] h_data;
    return 0;
}

//nvcc 02_Multiplicar_por_dos_multipleBloque.cu -o por2_mb