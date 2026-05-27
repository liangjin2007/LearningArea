# cuda中的纯头文件库

## 包含头文件
#include <cub/cub.cuh>

## 功能
- https://nvidia.github.io/cccl/unstable/cub/api_docs/thread_level.html
- https://nvidia.github.io/cccl/unstable/cub/api_docs/warp_wide.html
- https://nvidia.github.io/cccl/unstable/cub/api_docs/block_wide.html
- https://nvidia.github.io/cccl/unstable/cub/api_docs/device_wide.html

### Device wide 
- cub::DeviceScan
```
// Prefix Inclusive sum
y0 = x0
y1 = x0 + x1
...
cub::DeviceScan::InclusiveSum(
void *d_temp_storage,
size_t &temp_storage_bytes,
InputIteratorT d_in,
OutputIteratorT d_out,
NumItemsT num_items,
cudaStream_t stream = nullptr
)

// Prefix Exclusive Sum
y0 = 0
y1 = x0
y2 = x0 + x1
...
cub::DeviceScan::ExclusiveSum(
void *d_temp_storage,
size_t &temp_storage_bytes,
InputIteratorT d_in,
OutputIteratorT d_out,
NumItemsT num_items,
cudaStream_t stream = nullptr
)


// prefix inclusive scan
y0 = x0
y1 = scan_op(x0, x1)
...
cub::DeviceScan::InclusiveScan(
void *d_temp_storage,
size_t &temp_storage_bytes,
InputIteratorT d_in,
OutputIteratorT d_out,
ScanOpT scan_op,
InitValueT init_value,
NumItemsT num_items,
cudaStream_t stream = nullptr
)

// prefix exclusive scan
y0 = 0
y1 = x0
y2 = scan_op(x0, x1)
y3 = scan_op(scan_op(x0, x1), x2)
...
cub::DeviceScan::InclusiveScan(
void *d_temp_storage,
size_t &temp_storage_bytes,
InputIteratorT d_in,
OutputIteratorT d_out,
ScanOpT scan_op,
InitValueT init_value,
NumItemsT num_items,
cudaStream_t stream = nullptr
)
```

- cub::DeviceSegmentedReduce
```
cub::DeviceSegmentedReduce::Reduce(
void *d_temp_storage,
size_t &temp_storage_bytes,
InputIteratorT d_in,
OutputIteratorT d_out,
::cuda::std::int64_t num_segments,
BeginOffsetIteratorT d_begin_offsets,
EndOffsetIteratorT d_end_offsets,
ReductionOpT reduction_op,
T initial_value,
cudaStream_t stream = nullptr
)



cub::DeviceSegmentedReduce::Sum(
void *d_temp_storage,
size_t &temp_storage_bytes,
InputIteratorT d_in,
OutputIteratorT d_out,
::cuda::std::int64_t num_segments,
BeginOffsetIteratorT d_begin_offsets,
EndOffsetIteratorT d_end_offsets,
cudaStream_t stream = nullptr
)

cub::DeviceSegmentedReduce::Min
cub::DeviceSegmentedReduce::Max
cub::DeviceSegmentedReduce::ArgMin
cub::DeviceSegmentedReduce::ArgMax
```

- cub::DeviceRadixSort
```
// 回忆一下一般是怎么弄的 input sequence -> radix sort -> diff 得到1 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 ? 忘了

cub::DeviceRadixSort::SortPairs(
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


cub::DeviceRadixSort::SortPairsDescending(
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


cub::DeviceRadixSort::SortKeys(
void *d_temp_storage,
size_t &temp_storage_bytes,
const KeyT *d_keys_in,
KeyT *d_keys_out,
NumItemsT num_items,
int begin_bit = 0,
int end_bit = sizeof(KeyT) * 8,
cudaStream_t stream = nullptr
)

cub::DeviceRadixSort::SortKeysDescending(
void *d_temp_storage,
size_t &temp_storage_bytes,
const KeyT *d_keys_in,
KeyT *d_keys_out,
NumItemsT num_items,
int begin_bit = 0,
int end_bit = sizeof(KeyT) * 8,
cudaStream_t stream = nullptr
)
```


- cub::DeviceSelect
```
cub::DeviceSelect::Flagged(
void *d_temp_storage,
size_t &temp_storage_bytes,
InputIteratorT d_in,
FlagIterator d_flags,
OutputIteratorT d_out,
NumSelectedIteratorT d_num_selected_out,
::cuda::std::int64_t num_items,
cudaStream_t stream = nullptr
)



cub::DeviceSelect::Flagged(
void *d_temp_storage,
size_t &temp_storage_bytes,
InputIteratorT d_data,
FlagIterator d_flags,
NumSelectedIteratorT d_num_selected_out,
::cuda::std::int64_t num_items,
cudaStream_t stream = nullptr
)

// select_op is applied to d_flags
cub::DeviceSelect::FlaggedIf(
void *d_temp_storage,
size_t &temp_storage_bytes,
InputIteratorT d_in,
FlagIterator d_flags,
OutputIteratorT d_out,
NumSelectedIteratorT d_num_selected_out,
::cuda::std::int64_t num_items,
SelectOp select_op,
cudaStream_t stream = nullptr
)




cub::DeviceSelect::If(
void *d_temp_storage,
size_t &temp_storage_bytes,
InputIteratorT d_in,
OutputIteratorT d_out,
NumSelectedIteratorT d_num_selected_out,
::cuda::std::int64_t num_items,
SelectOp select_op,
cudaStream_t stream = nullptr
)



cub::DeviceSelect::Unique(
void *d_temp_storage,
size_t &temp_storage_bytes,
InputIteratorT d_in,
OutputIteratorT d_out,
NumSelectedIteratorT d_num_selected_out,
::cuda::std::int64_t num_items,
EqualityOpT equality_op,
cudaStream_t stream = nullptr
)


cub::DeviceSelect::UniqueByKey(
void *d_temp_storage,
size_t &temp_storage_bytes,
KeyInputIteratorT d_keys_in,
ValueInputIteratorT d_values_in,
KeyOutputIteratorT d_keys_out,
ValueOutputIteratorT d_values_out,
NumSelectedIteratorT d_num_selected_out,
NumItemsT num_items,
cudaStream_t stream = nullptr
)
```

-cub::DeviceRunLengthEncode
```
cub::DeviceRunLengthEncode::Encode(
void *d_temp_storage,
size_t &temp_storage_bytes,
InputIteratorT d_in,
UniqueOutputIteratorT d_unique_out,
LengthsOutputIteratorT d_counts_out,
NumRunsOutputIteratorT d_num_runs_out,
NumItemsT num_items,
cudaStream_t stream = nullptr
)


cub::DeviceRunLengthEncode::NonTrivialRuns(
void *d_temp_storage,
size_t &temp_storage_bytes,
InputIteratorT d_in,
OffsetsOutputIteratorT d_offsets_out,
LengthsOutputIteratorT d_lengths_out,
NumRunsOutputIteratorT d_num_runs_out,
NumItemsT num_items,
cudaStream_t stream = nullptr
)
```
