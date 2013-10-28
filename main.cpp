#include <iostream>
#include <ctime>       // clock_t, clock, CLOCKS_PER_SEC

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "kernel.h"


// works since CUDA 5.5, Nsight VS 3.1 which works with MSVS 2012 C++11
#if __cplusplus >= 201103L || _MSC_VER >= 1700
	#include <thread>
	using std::thread;
#else
	#include <boost/thread.hpp>
	using boost::thread;
#endif




unsigned char calculate_func(unsigned char const& val) {
	return hd_calculate_func(val);
}
// -----------------------------------------------

// get src data (emulate data-stream from other device - generate random data)
inline void get_data_from_other_device(unsigned char volatile*volatile dst_ptr, const unsigned int data_size) {
	//thrust::generate(dst_ptr, dst_ptr + data_size, rand);
	//thrust::fill(dst_ptr, dst_ptr + data_size, 1);
	const unsigned char simple_rand = rand();
	memset((void *)dst_ptr, simple_rand, data_size);
}
// -----------------------------------------------

bool compare_arrays(volatile unsigned char *const host_src_ptr, volatile unsigned char *const host_dst_ptr, const size_t c_array_size) {
	size_t i = 0;
	for(i = 0; i < c_array_size; ++i)
		if (host_src_ptr[i] != host_dst_ptr[i]) {
			std::cout << "find difference in: " << i << std::endl; 
			return false;
		}
	/*std::cout << "host_src_ptr & host_dst_ptr are equal " << std::endl;*/ 
	return true;
}
// ------------------------------------------------------------

void test_case(const unsigned int c_buff_size, const size_t MAX_BLOCKS_NUMBER) {
	// Set optimal numbers of THREADS, BLOCKS and array size

	// sizes of buffer and flag
	const unsigned int c_elements_number = (c_buff_size > 4096)?100*1024*1024 / c_buff_size
											: 5*1024*1024 / c_buff_size;			// must be multiple by 2
	const unsigned int c_array_size = c_buff_size * c_elements_number;	

	const size_t THREADS_NUMBER = (c_buff_size > 4*1024)? 1024 : c_buff_size / 4;	// scaling by threads
	size_t BLOCKS_NUMBER = c_buff_size / (4*THREADS_NUMBER);						// scaling by blocks
	if(BLOCKS_NUMBER > MAX_BLOCKS_NUMBER) BLOCKS_NUMBER = MAX_BLOCKS_NUMBER;

	//const size_t BLOCKS_NUMBER = 2;
	//const size_t THREADS_NUMBER = (c_buff_size > 4*1024*BLOCKS_NUMBER)? 1024 : c_buff_size / (4*BLOCKS_NUMBER);

	static const unsigned int c_flags_number = sizeof(bool) * BLOCKS_NUMBER;
	std::cout << "BLOCKS_NUMBER = " << BLOCKS_NUMBER << ", THREADS_NUMBER = " << THREADS_NUMBER << std::endl;	


	// init pointers
	// arrays: src & dst 
	unsigned char * host_src_array_ptr = NULL;
	unsigned char * host_dst_array_ptr = NULL;

	// src: temp buffer & flag
	unsigned char * host_src_buff_ptr = NULL;
	bool * host_src_flag_ptr = NULL;

	// dst: temp buffer & flag
	unsigned char * host_dst_buff_ptr = NULL;
	bool * host_dst_flag_ptr = NULL;

	// temp device memory
	unsigned char * dev_src_ptr1 = NULL;
	unsigned char * dev_src_ptr2 = NULL;
	unsigned char * dev_dst_ptr1 = NULL;
	unsigned char * dev_dst_ptr2 = NULL;



	// Allocate memory
	cudaHostAlloc(&host_src_array_ptr, c_array_size, cudaHostAllocMapped | cudaHostAllocPortable );
	cudaHostAlloc(&host_dst_array_ptr, c_array_size, cudaHostAllocMapped | cudaHostAllocPortable );

	cudaHostAlloc(&host_src_buff_ptr, c_buff_size, cudaHostAllocMapped | cudaHostAllocPortable );
	cudaHostAlloc(&host_src_flag_ptr, sizeof(*host_dst_flag_ptr)*c_flags_number, cudaHostAllocMapped | cudaHostAllocPortable );
	for(size_t i = 0; i < c_flags_number; ++i) 
		host_src_flag_ptr[i] = false;

	cudaHostAlloc(&host_dst_buff_ptr, c_buff_size, cudaHostAllocMapped | cudaHostAllocPortable );
	cudaHostAlloc(&host_dst_flag_ptr, sizeof(*host_dst_flag_ptr)*c_flags_number, cudaHostAllocMapped | cudaHostAllocPortable );
	for(size_t i = 0; i < c_flags_number; ++i) 
		host_dst_flag_ptr[i] = false;

	cudaMalloc(&dev_src_ptr1, c_buff_size);
	cudaMalloc(&dev_src_ptr2, c_buff_size);
	cudaMalloc(&dev_dst_ptr1, c_buff_size);
	cudaMalloc(&dev_dst_ptr2, c_buff_size);

	// generate random data
	thrust::fill(host_src_array_ptr, host_src_array_ptr + c_array_size, 0);
	thrust::fill(host_dst_array_ptr, host_dst_array_ptr + c_array_size, 0);
	

	std::cout << "each buff_size = " << c_buff_size << 
		", elements_number = " << c_elements_number << 
		", array_size = " << c_array_size/(1024*1024) << " MB" << std::endl;

	//std::cout << "--------------------------------------------------------------------" << std::endl;
	std::cout << std::endl;



	// =============================================================
	// Test-case CPU->GPU->CPU via DMA (overlaped)
	float c_time_dma_overlaped;

	{
		clock_t end, start;
		size_t iterations = 0;
		
		cudaStream_t stream1, stream2, stream3;
		cudaStreamCreate(&stream1); cudaStreamCreate(&stream2); cudaStreamCreate(&stream3);

		cudaDeviceSynchronize();
		start = clock();

		// step 1
		get_data_from_other_device(host_src_array_ptr, c_buff_size);	// emulate data-stream from other device 
		cudaMemcpy(dev_src_ptr1, host_src_array_ptr, c_buff_size, cudaMemcpyDefault);


		// overlaped: calculating in kernel-function and data transfer H->D & D->H
		for(size_t i = 0; i < c_elements_number; i+=2) {
			//std::cout << i << std::endl;

			// step 2
			k_dma(dev_dst_ptr1, dev_src_ptr1, c_buff_size, BLOCKS_NUMBER, THREADS_NUMBER, stream2);

			// step 3
			get_data_from_other_device(host_src_array_ptr + (i+1)*c_buff_size, c_buff_size);	// emulate data-stream from other device 
			cudaMemcpyAsync(dev_src_ptr2, host_src_array_ptr + (i+1)*c_buff_size, c_buff_size, cudaMemcpyDefault, stream3);	 
			cudaDeviceSynchronize();

			// step 1
			cudaMemcpyAsync(host_dst_array_ptr + i*c_buff_size, dev_dst_ptr1, c_buff_size, cudaMemcpyDefault, stream1);	 

 			// step 2
			k_dma(dev_dst_ptr2, dev_src_ptr2, c_buff_size, BLOCKS_NUMBER, THREADS_NUMBER, stream2);

			// step 3
			if(i+2 < c_elements_number) {
				get_data_from_other_device(host_src_array_ptr + (i+2)*c_buff_size, c_buff_size);	// emulate data-stream from other device 
				cudaMemcpyAsync(dev_src_ptr1, host_src_array_ptr + (i+2)*c_buff_size, c_buff_size, cudaMemcpyDefault, stream3);	  
			}
			cudaDeviceSynchronize();

			// step 1
			cudaMemcpyAsync(host_dst_array_ptr + (i+1)*c_buff_size, dev_dst_ptr2, c_buff_size, cudaMemcpyDefault, stream1);	 			
		}
		cudaDeviceSynchronize();

		


		++iterations;
		end = clock();

		c_time_dma_overlaped = (float)(end - start)/(CLOCKS_PER_SEC*iterations);
		std::cout << "DMA-overlaped:\t time: " << c_time_dma_overlaped << 
			", " << c_array_size/(c_time_dma_overlaped * 1024*1024) << " MB/sec" << 
			", 3 x avg-lat: " << 3*c_time_dma_overlaped/c_elements_number << std::endl;


		// calculate on host
		for(size_t i = 0; i < c_array_size; ++i)
			host_src_array_ptr[i] = calculate_func(host_src_array_ptr[i]);

		// Compare data
		compare_arrays(host_src_array_ptr, host_dst_array_ptr, c_array_size);
	}

	//int c; std::cin >> c;



	// =============================================================
	// Test-case CPU->GPU->CPU via DMA (sequentially)
	float c_time_dma_sequentially;

	{
		clock_t end, start;
		size_t iterations = 0;
		
		cudaStream_t stream1;
		cudaStreamCreate(&stream1);

		cudaDeviceSynchronize();
		start = clock();

		// sequentially: calculating in kernel-function and data transfer H->D & D->H
		for(size_t i = 0; i < c_elements_number; ++i) {
			//std::cout << i << std::endl;

			// step 1
			get_data_from_other_device(host_src_array_ptr + i*c_buff_size, c_buff_size);	// emulate data-stream from other device 
			cudaMemcpy(dev_src_ptr1, host_src_array_ptr + i*c_buff_size, c_buff_size, cudaMemcpyDefault);
			cudaDeviceSynchronize();

			// step 2
			k_dma(dev_dst_ptr1, dev_src_ptr1, c_buff_size, BLOCKS_NUMBER, THREADS_NUMBER, stream1);
			cudaDeviceSynchronize();

			// step 3
			// accumulate results
			cudaMemcpy(host_dst_array_ptr + i*c_buff_size, dev_dst_ptr1, c_buff_size, cudaMemcpyDefault);	 
		
		}
		++iterations;
		end = clock();

		c_time_dma_sequentially = (float)(end - start)/(CLOCKS_PER_SEC*iterations);
		std::cout << "DMA-seq:\t time: " << c_time_dma_sequentially << 
			", " << c_array_size/(c_time_dma_sequentially * 1024*1024) << " MB/sec" << 
			", avg latency: " << c_time_dma_sequentially/c_elements_number << 
			" (1 X)" << std::endl;


		// calculate on host
		for(size_t i = 0; i < c_array_size; ++i)
			host_src_array_ptr[i] = calculate_func(host_src_array_ptr[i]);

		// Compare data
		compare_arrays(host_src_array_ptr, host_dst_array_ptr, c_array_size);
	}



	// =============================================================
	// Test-case CPU->GPU->CPU via UVA (with temp buffer)
	{
		clock_t end, start;
		size_t iterations = 0;

		cudaStream_t stream1;
		cudaStreamCreate(&stream1);

		const_cast<volatile bool *>(host_src_flag_ptr)[0] = false;
		const_cast<volatile bool *>(host_dst_flag_ptr)[0] = false;

		// src: temp buffer & flag
		unsigned char * uva_src_buff_ptr = NULL;
		bool * uva_src_flag_ptr = NULL;

		// dst: temp buffer & flag
		unsigned char * uva_dst_buff_ptr = NULL;
		bool * uva_dst_flag_ptr = NULL;
		
		// host_ptr -> uva_ptr
		cudaHostGetDevicePointer(&uva_src_buff_ptr, host_src_buff_ptr, 0);
		cudaHostGetDevicePointer(&uva_src_flag_ptr, host_src_flag_ptr, 0);

		cudaHostGetDevicePointer(&uva_dst_buff_ptr, host_dst_buff_ptr, 0);		
		cudaHostGetDevicePointer(&uva_dst_flag_ptr, host_dst_flag_ptr, 0);		
		
		cudaDeviceSynchronize();
		
		const bool init_flag = const_cast<volatile bool *>(host_src_flag_ptr)[0] = false;
		
		// in separate thread launch permanent scan on GPU (need if doesn't use TCC-driver)
		thread t1( [&] () {
			cudaStream_t stream1;
			cudaStreamCreate(&stream1);

			k_uva_1b(uva_dst_flag_ptr, uva_src_flag_ptr, uva_dst_buff_ptr, uva_src_buff_ptr, c_buff_size, init_flag, 
				c_elements_number, 1, THREADS_NUMBER, stream1);
			cudaDeviceSynchronize();
		} );

		start = clock();
		for(size_t i = 0; i < c_elements_number; ++i) {
			//std::cout << i << std::endl;

			get_data_from_other_device(host_src_buff_ptr, c_buff_size);	// emulate data-stream from other device 

			// set query flag
			const_cast<volatile bool *>(host_src_flag_ptr)[0] = !const_cast<volatile bool *>(host_src_flag_ptr)[0];
			//no need _mm_sfence(); because uses cache-snooping and doesn't use WriteCombined marked cache 

			// for a reference comparison
			memcpy(host_src_array_ptr + i*c_buff_size, host_src_buff_ptr, c_buff_size);

			// wait answer flag
			while (const_cast<volatile bool *>(host_dst_flag_ptr)[0] != const_cast<volatile bool *>(host_src_flag_ptr)[0]) {
				//_mm_lfence();
			}

			// accumulate results
			memcpy(host_dst_array_ptr + i*c_buff_size, host_dst_buff_ptr, c_buff_size);
		}
		++iterations;
		end = clock();


		const float c_time_uva_calc = (float)(end - start)/(CLOCKS_PER_SEC*iterations);
		std::cout.precision(3);
		std::cout.width(9);
		std::cout << "UVA-1B: \t time: " << c_time_uva_calc << 
			", " << c_array_size/(c_time_uva_calc * 1024*1024) << " MB/sec" << 
			", avg latency: " << c_time_uva_calc/c_elements_number << 
			" (" << (c_time_dma_sequentially) / (c_time_uva_calc) << " X)" << std::endl;

		t1.join();


		// calculate on host
		for(size_t i = 0; i < c_array_size; ++i)
			host_src_array_ptr[i] = calculate_func(host_src_array_ptr[i]);

		// Compare data
		compare_arrays(host_src_array_ptr, host_dst_array_ptr, c_array_size);


		//std::cout << "UVA-1B faster than DMA-seq in: " << (c_time_dma_sequentially) / (c_time_uva_calc) <<
			//" X times" << std::endl;
	}


	// =============================================================
	// Test-case CPU->GPU->CPU via UVA 
	{
		clock_t end, start;
		size_t iterations = 0;

		cudaStream_t stream1;
		cudaStreamCreate(&stream1);

		const_cast<volatile bool *>(host_src_flag_ptr)[0] = false;
		const_cast<volatile bool *>(host_dst_flag_ptr)[0] = false;

		// src & dst uva arrays
		unsigned char * uva_src_array_ptr = NULL;
		unsigned char * uva_dst_array_ptr = NULL;

		// src: temp buffer & flag
		unsigned char * uva_src_buff_ptr = NULL;
		bool * uva_src_flag_ptr = NULL;

		// dst: temp buffer & flag
		unsigned char * uva_dst_buff_ptr = NULL;
		bool * uva_dst_flag_ptr = NULL;
		
		// host_ptr -> uva_ptr
		cudaHostGetDevicePointer(&uva_src_array_ptr, host_src_array_ptr, 0);
		cudaHostGetDevicePointer(&uva_dst_array_ptr, host_dst_array_ptr, 0);

		cudaHostGetDevicePointer(&uva_src_flag_ptr, host_src_flag_ptr, 0);
		cudaHostGetDevicePointer(&uva_dst_flag_ptr, host_dst_flag_ptr, 0);		

		cudaDeviceSynchronize();
		
		const bool init_flag = const_cast<volatile bool *>(host_src_flag_ptr)[0] = false;
		
		// in separate thread launch permanent scan on GPU (need if doesn't use TCC-driver)
		thread t1( [&] () {
			cudaStream_t stream1;
			cudaStreamCreate(&stream1);

			k_uva(uva_dst_flag_ptr, uva_src_flag_ptr, uva_dst_array_ptr, uva_src_array_ptr, c_buff_size, init_flag, 
				c_elements_number, BLOCKS_NUMBER, THREADS_NUMBER, stream1);
			cudaDeviceSynchronize();
		} );

		start = clock();
		for(size_t i = 0; i < c_elements_number; ++i) {
			//std::cout << i << std::endl;

			// set query flag
			const bool current_flag = const_cast<volatile bool *>(host_src_flag_ptr)[0] = !const_cast<volatile bool *>(host_src_flag_ptr)[0];
			//no need _mm_sfence(); because uses cache-snooping and doesn't use WriteCombined marked cache 

			// generate new data
			get_data_from_other_device(host_src_buff_ptr, c_buff_size);	// emulate data-stream from other device 
			//_mm_sfence();
			
			// optimized spin-wait (speedup about 3%) - wait answer flag for all block_id
			switch(BLOCKS_NUMBER) {
				case 8:
					if(current_flag) while (((volatile uint64_t *)host_dst_flag_ptr)[0] == 0);
					else while (((volatile uint64_t *)host_dst_flag_ptr)[0] != 0);
				break;
				case 4:
					if(current_flag) while (((volatile uint32_t *)host_dst_flag_ptr)[0] == 0);
					else while (((volatile uint32_t *)host_dst_flag_ptr)[0] != 0);
				break;
				case 2:
					if(current_flag) while (((volatile uint16_t *)host_dst_flag_ptr)[0] == 0);
					else while (((volatile uint16_t *)host_dst_flag_ptr)[0] != 0);
				break;
				case 1:
					if(current_flag) while (((volatile uint8_t *)host_dst_flag_ptr)[0] == 0);
					else while (((volatile uint8_t *)host_dst_flag_ptr)[0] != 0);
				break;
				default:
					for(size_t block_id = 0; block_id < BLOCKS_NUMBER; ++block_id) {
						// wait answer flag for current block_id
						while (const_cast<volatile bool *>(host_dst_flag_ptr)[block_id] != current_flag) {}
					}
				break;
			}			
		}
		++iterations;
		end = clock();


		const float c_time_uva_calc = (float)(end - start)/(CLOCKS_PER_SEC*iterations);
		std::cout.precision(3);
		std::cout.width(9);
		std::cout << "UVA-MB: \t time: " << c_time_uva_calc << 
			", " << c_array_size/(c_time_uva_calc * 1024*1024) << " MB/sec" << 
			", avg latency: " << c_time_uva_calc/c_elements_number << 
			" (" << (c_time_dma_sequentially) / (c_time_uva_calc) << " X)" << std::endl;

		t1.join();


		// calculate on host
		for(size_t i = 0; i < c_array_size; ++i)
			host_src_array_ptr[i] = calculate_func(host_src_array_ptr[i]);

		// Compare data
		compare_arrays(host_src_array_ptr, host_dst_array_ptr, c_array_size);


		//std::cout << "UVA-MB faster than DMA-seq in: " << (c_time_dma_sequentially) / (c_time_uva_calc) <<
			//" X times" << std::endl;
	}

	//for(size_t i = 0; i < 100; ++i)
		//std::cout << (unsigned int)host_dst_array_ptr[i] << ", ";

	std::cout << "--------------------------------------------------------------------" << std::endl;
}
// ------------------------------------------------------------

int main() {

	srand (time(NULL));

	// count devices & info
	int device_count;
	cudaDeviceProp device_prop;

	// get count Cuda Devices
	cudaGetDeviceCount(&device_count);
	std::cout << "Device count: " <<  device_count << std::endl;

	if (device_count > 100) device_count = 0;
	for (int i = 0; i < device_count; i++)
	{
		// get Cuda Devices Info
		cudaGetDeviceProperties(&device_prop, i);
		std::cout << "Device" << i << ": " <<  device_prop.name;
		std::cout << " (" <<  device_prop.totalGlobalMem/(1024*1024) << " MB)";
		std::cout << ", CUDA capability: " <<  device_prop.major << "." << device_prop.minor << std::endl;	
		std::cout << "UVA: " <<  device_prop.unifiedAddressing << std::endl;
		std::cout << "canMapHostMemory: " <<  device_prop.canMapHostMemory  << std::endl;
		std::cout << "MAX BLOCKS NUMBER: " <<  device_prop.maxGridSize[0] << std::endl;
		std::cout << "tccDriver: " <<  device_prop.tccDriver << std::endl;
		std::cout << "multiProcessorCount: " <<  device_prop.multiProcessorCount << std::endl;
	}
	std::cout << std::endl;
	std::cout << "CPU-RAM -> GPU(calculate) -> CPU-RAM" << std::endl;
	//std::cout << "__cplusplus = " << __cplusplus << std::endl;	

	if (device_count >= 0) {
		cudaSetDevice(0);
		std::cout << "Selected Device" << 0 << std::endl;
	}
	else {
		std::cout << "Error: device_count = " << device_count << std::endl;
		system ("PAUSE");
		return 0;
	}

	std::cout << std::endl;
	std::cout << "DMA - copy by using DMA-controller and launch kernel-function for each packet" << std::endl;
	std::cout << "UVA - copy by using Unified Virtual Addressing and launch kernel-function once" << std::endl;
	std::cout << "--------------------------------------------------------------------" << std::endl;
	std::cout << std::endl;
		



	cudaGetDeviceProperties(&device_prop, 0);
	const size_t MAX_BLOCKS_NUMBER = 2; //device_prop.multiProcessorCount;
	

	// Can Host map memory
	cudaSetDeviceFlags(cudaDeviceMapHost);


	// test cases (different block size)
	for(size_t i = 64; i <= 4*65536; i *= 2) {
		test_case(i, MAX_BLOCKS_NUMBER);			
	}


	int b;
	std::cin >> b;

	return 0;
}