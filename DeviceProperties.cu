#include<stdio.h>
#include<cuda.h>


int main(void) {

	// function declaration
	void printCudaDeviceProperties();

	// code
	printCudaDeviceProperties();


}

void printCudaDeviceProperties() {

	// code
	printf("CUDA Information...!!\n");

	printf("=====================\n");

	cudaError_t ret_cuda_rt;

	int devCount;
	ret_cuda_rt = cudaGetDeviceCount(&devCount);
	if (ret_cuda_rt != cudaSuccess) {
		
		printf("Cuda Runtime API Error - cudaDeviceCount() failed.");
	}
	else if (devCount == 0) {
		printf("There is no cuda supported device on this system.");
		return;
	}
	else {
		printf("Total number of CUDA supporting GPU Device(s) on this system : %d\n", devCount);
		for (int i = 0; i < devCount; i++) {
			cudaDeviceProp devProp;
			int driverVersion = 0, runtimeVersion = 0;
			ret_cuda_rt = cudaGetDeviceProperties(&devProp, i);
			
			if (ret_cuda_rt != cudaSuccess) {
			
				printf("%s in %s at line %d\n", cudaGetErrorString(ret_cuda_rt), __FILE__, __LINE__);
			
				return;
			}
			printf("\n");
			cudaDriverGetVersion(&driverVersion);
			cudaRuntimeGetVersion(&runtimeVersion);

			printf("***** Cuda driver and Runtime info ****\n");
			printf("=======================================\n");
			printf("Cuda driver version   : %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
			printf("Cuda Runtime version  : %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
			printf("\n");
			printf("=======================================\n");
			printf("***** GPU General Information ****\n");
			printf("=======================================\n");
			printf("GPU Device Number : %d\n", i);
			printf("GPU Device Name :%s\n", devProp.name);
			printf("GPU Device Compute Capability :%d.%d\n", devProp.major, devProp.minor);
			printf("GPU Clock Rate : %d\n", devProp.clockRate);
			printf("GPU Device Type : ");
			if (devProp.integrated) 
				printf("Integrated (On-board) \n");
			else
				printf("Discrete (Card)\n");

			printf("\n");
			printf("=======================================\n");
			printf("***** GPU Device Memory Information ****\n");
			printf("=======================================\n");
			printf("GPU Device Total Memory GB = %.0f MB =%llu Bytes\n", ((float) devProp.totalGlobalMem / 1048576.0f) / 1024.0f, (unsigned long long) devProp.totalGlobalMem);
			printf("GPU Device Constant Memory   :%lu Bytes \n", (unsigned long)devProp.totalConstMem);
			printf("GPU Device Shared Memory Per SMProcessor     :%lu\n", (unsigned long)devProp.sharedMemPerBlock);
			printf("\n");
			printf("=======================================\n");
			printf("***** GPU Device Multiprocessor Information ****\n");
			printf("=======================================\n");
			printf("GPU Device Number of SMProcessors :%d\n", devProp.multiProcessorCount);
			printf("GPU Device Numner of Registers Per SMProcessor :%d\n", devProp.regsPerBlock);
			printf("=======================================\n");
			printf("***** GPU Device Thread Information ****\n");
			printf("=======================================\n");
			printf("GPU Device Maximum Number of Threads Per Block :%d\n", devProp.maxThreadsPerMultiProcessor);
			printf("GPU Device Maximum Threads in Warp		  : %d\n", devProp.warpSize);
			printf("GPU Device Maximum Thread Dimensions      : (%d, %d, %d )\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
			printf("GPU Device Maximum Thread Grid Dimensions : (%d, %d, %d)\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
			printf("\n");
			printf("=======================================\n");
			printf("***** GPU Device Driver Information ****\n");
			printf("=======================================\n");
			printf("GPU Device has ECC (Err. Correction Code) support :%s\n", devProp.ECCEnabled ? "Enabled" : "Disabled");
#if  defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
			printf("GPU Device CUDA Driver Mode (TCC or WDDM)  : %s\n", devProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
			printf("****************************************\n");
		
			}
		}
	}
