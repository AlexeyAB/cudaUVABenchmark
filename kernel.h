
/// Return calculated value (test function)
unsigned char hd_calculate_func(unsigned char const& val);
// -----------------------------------------------


/// Uses only GPU-RAM (with prior copying data CPU-RAM -> GPU-RAM and after copying back by using DMA-controller)
///
/// @param dst_ptr - pointer to destination temporary buffer in GPU-RAM for calculated data
/// @param src_ptr - pointer to source temporary buffer in GPU-RAM for data to calculate
/// @param c_buff_size - size of source data
///
void k_dma(unsigned char *const dst_ptr, unsigned char *const src_ptr, const unsigned int c_buff_size,
		   const size_t BLOCKS, const size_t THREADS, cudaStream_t &stream);
// -----------------------------------------------


/// Uses UVA and only CPU-RAM (without GPU-RAM) with src & dst temporary buffers
///
/// Permanent scan of source's flag, when it ready then copy data from the source temporary buffer to registers of GPU-Cores, 
/// calculate, and then store to the destination temporary buffer
///
/// @param uva_dst_flag_ptr - pointer to destination flag in CPU-RAM which switch when GPU-data ready (calculate finished)
/// @param uva_src_flag_ptr - pointer to source flag in CPU-RAM which switch when CPU-data ready (ready to calculate)
/// @param uva_dst_buff_ptr - pointer to destination temporary buffer in CPU-RAM for calculated data
/// @param uva_src_buff_ptr - pointer to source temporary buffer in CPU-RAM for data to calculate
/// @param c_buff_size - size of source data
/// @param init_flag - initial value for source flag located by pointer - uva_src_flag_ptr
/// @param iterations - number of iteration for permanent scan before exit from kernel-function
///
void k_uva_1b(volatile bool * uva_dst_flag_ptr, volatile bool * uva_src_flag_ptr,
			volatile unsigned char * uva_dst_buff_ptr, volatile unsigned char * uva_src_buff_ptr, const unsigned int c_buff_size, 
			const bool init_flag, const unsigned int iterations, 
			const size_t BLOCKS, const size_t THREADS, cudaStream_t &stream);
// -----------------------------------------------


/// Uses UVA and only CPU-RAM (without GPU-RAM) with current part of src & dst arrays
///
/// Permanent scan of source's flag, when it ready then copy data from the current part of source array to registers of GPU-Cores, 
/// calculate, and then store to the current part of destination array
///
/// @param uva_dst_flag_ptr - pointer to destination flag in CPU-RAM which switch when GPU-data ready (calculate finished)
/// @param uva_src_flag_ptr - pointer to source flag in CPU-RAM which switch when CPU-data ready (ready to calculate)
/// @param uva_dst_buff_ptr - pointer to destination array in CPU-RAM for calculated data
/// @param uva_src_buff_ptr - pointer to source array in CPU-RAM for data to calculate
/// @param c_buff_size - size of source data
/// @param init_flag - initial value for source flag located by pointer - uva_src_flag_ptr
/// @param iterations - number of iteration for permanent scan before exit from kernel-function
///

void k_uva(volatile bool * uva_dst_flag_ptr, volatile bool * uva_src_flag_ptr,
			volatile unsigned char * uva_dst_buff_ptr, volatile unsigned char * uva_src_buff_ptr, const unsigned int c_buff_size, 
			const bool init_flag, const unsigned int iterations, 
			const size_t BLOCKS, const size_t THREADS, cudaStream_t &stream);


