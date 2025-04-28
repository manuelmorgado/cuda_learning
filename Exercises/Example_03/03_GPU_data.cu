#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "Número de GPUs disponibles: " << device_count << std::endl;

    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        std::cout << "\n=== GPU " << device << " ===" << std::endl;
        std::cout << "Nombre: " << prop.name << std::endl;
        std::cout << "Memoria Global total: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Multiprocesadores (SMs): " << prop.multiProcessorCount << std::endl;
        std::cout << "Máximo número de hilos por bloque: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Tamaño máximo de un bloque de hilos: ("
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "Tamaño máximo de una grilla: ("
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << ")" << std::endl;
        std::cout << "Memoria compartida por bloque: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "Máxima cantidad de registros por bloque: " << prop.regsPerBlock << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Tamaño de warp: " << prop.warpSize << std::endl;
        std::cout << "Reloj de GPU: " << prop.clockRate / 1000 << " MHz" << std::endl;
    }

    return 0;
}

//nvcc 03_GPU_data.cu -o my_gpu