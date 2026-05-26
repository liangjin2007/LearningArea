# cuda中的纯头文件库

## 包含头文件
#include <cub/cub.cuh>

## 功能
- https://nvidia.github.io/cccl/unstable/cub/api_docs/thread_level.html
- https://nvidia.github.io/cccl/unstable/cub/api_docs/warp_wide.html
- https://nvidia.github.io/cccl/unstable/cub/api_docs/block_wide.html
- https://nvidia.github.io/cccl/unstable/cub/api_docs/device_wide.html

- Device wide 
```

// Prefix sum
template<typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
static inline cudaError_t ExclusiveSum(
void *d_temp_storage,
size_t &temp_storage_bytes,
InputIteratorT d_in,
OutputIteratorT d_out,
NumItemsT num_items,
cudaStream_t stream = nullptr
)

template<typename KeyT, typename ValueT, typename NumItemsT>
static inline cudaError_t SortPairs(
void *d_temp_storage,
size_t &temp_storage_bytes,
const KeyT *d_keys_in,
KeyT *d_keys_out,
const ValueT *d_values_in,
ValueT *d_values_out,
NumItemsT num_items,
int begin_bit = 0,
int end_bit = sizeof(KeyT) * 8,
cudaStream_t stream = nullptr
)

template<typename InputIteratorT, typename FlagIterator, typename OutputIteratorT, typename NumSelectedIteratorT, ::cuda::std::enable_if_t<!::cuda::std::is_integral_v<NumSelectedIteratorT>, int> = 0>
static inline cudaError_t Flagged(
void *d_temp_storage,
size_t &temp_storage_bytes,
InputIteratorT d_in,
FlagIterator d_flags,
OutputIteratorT d_out,
NumSelectedIteratorT d_num_selected_out,
::cuda::std::int64_t num_items,
cudaStream_t stream = nullptr
)
```
