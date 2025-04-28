#include <iostream>
#include <vector>
#include <cuda.h>

#define CHECK_CUDA(call) \
    if ((call) != CUDA_SUCCESS) { \
        std::cerr << "Error en llamada CUDA Driver API en " << __FILE__ << ":" << __LINE__ << std::endl; \
        return -1; \
    }

int main() {

	CUdevice cuDevice;
	CUcontext cuContext;
	CUmodule cuModule;
	CUfunction cuFunction;

	const int N = 10;
	std::vector<int> h_datos(N);

	for (int i = 0; i < N; ++i){
		h_datos[i] = i;
	}

	// Inicializa el Driver API
	CHECK_CUDA(cuInit(0));
	CHECK_CUDA(cuDeviceGet(&cuDevice, 0));
	CHECK_CUDA(cuCtxCreate(&cuContext, 0, cuDevice));

	// Cargar module (el PTX compilado)
	CHECK_CUDA(cuModuleLoad(&cuModule, "multiplicar_por_dos_kernel.ptx"));
	CHECK_CUDA(cuModuleGetFunction(&cuFunction, cuModule, "multiplicar_por_dos"));

	// Reservar memoria en GPU
	CUdeviceptr d_datos;
	CHECK_CUDA(cuMemAlloc(&d_datos, N * sizeof(int)));

	// Copiar datos del host al device
	CHECK_CUDA(cuMemcpyHtoD(d_datos, h_datos.data(), N * sizeof(int)));

	// Perparar parametros para el kernel
	void* args[] = { (void*)&d_datos, (void*)&N };
	
	int threads_per_block = 256;
	int blocks = (N + threads_per_block - 1)/ threads_per_block;

	// Lanzar el kernel
	CHECK_CUDA(cuLaunchKernel(cuFunction,
							  blocks, 1, 1,				// gridDim
							  threads_per_block, 1, 1,	// blockDim
							  0,						// shared memory
							  0,						// stream
							  args,						// argumentos
							  nullptr                   // extra
							  ));

	// Esperar a que termine el kernel
	CHECK_CUDA(cuCtxSynchronize());

	// Copiar de vuelta del device al host
	CHECK_CUDA(cuMemcpyDtoH(h_datos.data(), d_datos, N * sizeof(int)));

	// Imprimir resultados
	std::cout << "Resultados:\n";
	for  (auto val : h_datos){
		std::cout << val << " ";
	}
	std::cout << std::endl;

	// Liberar memoria
	cuMemFree(d_datos);
	cuModuleUnload(cuModule);
	cuCtxDestroy(cuContext);

	return 0;
}

// nvcc 04_Multiplicar_por_dos_driver_api.cu -lcuda -o driver_api