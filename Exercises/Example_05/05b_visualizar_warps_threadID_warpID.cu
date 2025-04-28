#include <iostream>
#include <cuda.h>

__global__ void kernel_guardar_warp_info(int* salida_tid, int* salida_warp) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;  // ID global del hilo
    int warp_id = tid / 32;                           // Warp ID: cada 32 hilos forman un warp

    salida_tid[tid] = tid;       // Guardar el Thread ID
    salida_warp[tid] = warp_id;  // Guardar el Warp ID
}

int main() {
    const int N = 64;  // Número total de hilos que queremos lanzar

    int threads_per_block = 32;  // Cambia este valor si quieres experimentar
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    int* h_tid = new int[N];
    int* h_warp = new int[N];

    int *d_tid, *d_warp;
    cudaMalloc(&d_tid, N * sizeof(int));
    cudaMalloc(&d_warp, N * sizeof(int));

    kernel_guardar_warp_info<<<blocks, threads_per_block>>>(d_tid, d_warp);

    cudaDeviceSynchronize();

    cudaMemcpy(h_tid, d_tid, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_warp, d_warp, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Thread ID - Warp ID:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "Hilo " << h_tid[i] << " pertenece al Warp " << h_warp[i] << "\n";
    }

    delete[] h_tid;
    delete[] h_warp;
    cudaFree(d_tid);
    cudaFree(d_warp);

    return 0;
}

// nvcc 05b_visualizar_warps_threadID_warpID.cu -o visualizar_warps_ordenado