


#include "kernel.h"


// Test calculate-function both for CPU and GPU
template<typename T>
inline __host__ __device__ 
T calculate_func(T const& val) {
	return val*val*2;
}
// -----------------------------------------------

// High performance computing using only registers (no shared memory)
inline __host__ __device__ 
unsigned int calculate_with_shift(unsigned int src) {
	unsigned int dst = 0;
	#pragma unroll
	for(int i = 0; i < 4; ++i) {
		unsigned int tmp =  calculate_func<unsigned char>( src & 0xFF );
		dst |= tmp << (8*i);
		src >>= 8;
	}
	return dst;
}
// -----------------------------------------------

/// Optimally coalescing access to RAM by 128 bytes for each WARP, and then store into registers of GPU-Cores
__device__  
inline void copy_transform(volatile unsigned char * dst_buff_ptr, volatile unsigned char * src_buff_ptr,
						   const unsigned int &c_buff_size)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// to use uint32_t for coalescing access to the GPU-RAM or CPU-RAM 
	for(int k = 0; k < c_buff_size/sizeof(unsigned int); k += blockDim.x * gridDim.x) {
			reinterpret_cast<volatile unsigned int *>(dst_buff_ptr)[tid + k] = 
				calculate_with_shift( reinterpret_cast<volatile unsigned int *>(src_buff_ptr)[tid + k] );
	}
}
// -----------------------------------------------

// Uses only GPU-RAM with prior copying data by using DMA-controller
__global__ void kernel_function_dma(unsigned char *const dst_ptr, unsigned char *const src_ptr, const unsigned int c_buff_size) {

	copy_transform(dst_ptr, src_ptr, c_buff_size);
}
// -----------------------------------------------

// Uses UVA and only CPU-RAM (without GPU-RAM) with temporary buffer
__global__
void kernel_uva_1b(volatile bool * uva_dst_flag_ptr, volatile bool * uva_src_flag_ptr, 
					volatile unsigned char * uva_dst_buff_ptr, volatile unsigned char * uva_src_buff_ptr, 
					const unsigned int c_buff_size,
					const bool init_flag, const unsigned int iterations)
{
	bool current_flag = init_flag;

	for(unsigned int i = 0; i < iterations; ++i) 
	{		
		// wait for src_flag
		if(threadIdx.x == 0) 
		{
			// spin wait when src_flag_ptr[0] and current_flag will be different
			while(uva_src_flag_ptr[0] == current_flag) {   }
			current_flag = !current_flag;	
		}
		__syncthreads();

		// copy data with transformation (with temporary buffer)
		copy_transform(uva_dst_buff_ptr, uva_src_buff_ptr, c_buff_size);

		__syncthreads();

		// set dst_flag
		if(threadIdx.x == 0) {
			uva_dst_flag_ptr[blockIdx.x] = current_flag;	// make that dst_flag_ptr[blockIdx.x] and current_flag will be equal
			__threadfence_system();						// sync with CPU-RAM
		}
			
	}
}
// -----------------------------------------------

// Uses UVA and only CPU-RAM (without GPU-RAM) with current part of array
__global__
void kernel_uva(volatile bool * uva_dst_flag_ptr, volatile bool * uva_src_flag_ptr, 
					volatile unsigned char * uva_dst_buff_ptr, volatile unsigned char * uva_src_buff_ptr, 
					const unsigned int c_buff_size,
					const bool init_flag, const unsigned int iterations)
{
	bool current_flag = init_flag;

	for(unsigned int i = 0; i < iterations; ++i) 
	{		
		// wait for src_flag
		if(threadIdx.x == 0) 
		{
			// spin wait when src_flag_ptr[0] and current_flag will be different
			while(uva_src_flag_ptr[0] == current_flag) {   }
			current_flag = !current_flag;	
		}
		__syncthreads();

		// copy data with transformation (without temporary buffer)
		copy_transform(uva_dst_buff_ptr + i*c_buff_size, uva_src_buff_ptr + i*c_buff_size, c_buff_size);

		__threadfence_system();
		__syncthreads();

		// set dst_flag
		if(threadIdx.x == 0) {
			uva_dst_flag_ptr[blockIdx.x] = current_flag;	// make that dst_flag_ptr[blockIdx.x] and current_flag will be equal
			__threadfence_system();						// sync with CPU-RAM
		}

	}
}
// -----------------------------------------------
// =============================================================
// Wrappers CUDA C++ -> C++:
// =============================================================

// Return calculated value
unsigned char hd_calculate_func(unsigned char const& val) {
	return calculate_func(val);
}
// -----------------------------------------------

// Uses only GPU-RAM with prior copying data by using DMA-controller
void k_dma(unsigned char *const dst_ptr, unsigned char *const src_ptr, const unsigned int c_buff_size,
		   const size_t BLOCKS, const size_t THREADS, cudaStream_t &stream)
{
	kernel_function_dma<<<BLOCKS, THREADS, 0, stream>>>(dst_ptr, src_ptr, c_buff_size);
}
// -----------------------------------------------

// Uses UVA and only CPU-RAM (without GPU-RAM) with temporary buffer
void k_uva_1b(volatile bool * uva_dst_flag_ptr, volatile bool * uva_src_flag_ptr,
			volatile unsigned char * uva_dst_buff_ptr, volatile unsigned char * uva_src_buff_ptr, const unsigned int c_buff_size, 
			const bool init_flag, const unsigned int iterations, 
			const size_t BLOCKS, const size_t THREADS, cudaStream_t &stream) 
{
	kernel_uva_1b<<<BLOCKS, THREADS, 0, stream>>>(uva_dst_flag_ptr, uva_src_flag_ptr, uva_dst_buff_ptr, uva_src_buff_ptr, 
												c_buff_size, init_flag, iterations);
}
// -----------------------------------------------

// Uses UVA and only CPU-RAM (without GPU-RAM) with current part of array
void k_uva(volatile bool * uva_dst_flag_ptr, volatile bool * uva_src_flag_ptr,
			volatile unsigned char * uva_dst_buff_ptr, volatile unsigned char * uva_src_buff_ptr, const unsigned int c_buff_size, 
			const bool init_flag, const unsigned int iterations, 
			const size_t BLOCKS, const size_t THREADS, cudaStream_t &stream) 
{
	kernel_uva<<<BLOCKS, THREADS, 0, stream>>>(uva_dst_flag_ptr, uva_src_flag_ptr, uva_dst_buff_ptr, uva_src_buff_ptr, 
												c_buff_size, init_flag, iterations);
}
// -----------------------------------------------

