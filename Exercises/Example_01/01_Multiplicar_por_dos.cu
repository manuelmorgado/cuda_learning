#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Kernel CUDA
__global__ void multiplicar_por_dos(int *datos, int N){
    int idx = threadIdx.x;
    if (idx < N) {
        datos[idx] *= 2;
    }
}

// Funcion principal
int main() {

    // Mide memoria libre
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Memoria libre: " << free_mem / 1024 / 1024 << " MB\n";

    const int N = 1000000;
    std::vector<int> h_datos(N);

    // Inicializa los elementos del vector
    for (int i = 0; i<N; ++i)
        h_datos[i] = i;

    // Reservar el bloque de memoria en la GPU
    int *d_datos;
    cudaMalloc(&d_datos, N * sizeof(int));

    // Mueve memory al GPU (device)
    cudaMemcpy(d_datos, h_datos.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Lanza kernel
    multiplicar_por_dos<<< 1, N >>>(d_datos, N);

    // Mueve de vuelta la memoria al CPU (host)
    cudaMemcpy(h_datos.data(), d_datos, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Muestra los resultados
    for (auto val : h_datos){
        std::cout << val << " ";
    }

    std::cout << std::endl;

    // Mide memoria libre
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Memoria libre: " << free_mem / 1024 / 1024 << " MB\n";

    // Liberar memoria
    cudaFree(d_datos);

    // Mide memoria libre
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Memoria libre: " << free_mem / 1024 / 1024 << " MB\n";

    return 0;


}

//nvcc 01_Multiplicar_por_dos.cu -o por2