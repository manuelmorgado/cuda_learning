extern "C" __global__ void multiplicar_por_dos(int *datos, int N){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N){
		datos[idx] *= 2;
	}
}

// nvcc -ptx multiplicar_por_dos_kernel.cu -o multiplicar_por_dos_kernel.ptx