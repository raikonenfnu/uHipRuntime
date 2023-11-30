#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <mutex>
#include "utils.h"
#include <chrono>

extern "C" __device__ __attribute__((const)) half __ockl_wfred_max_f16(half);

using float16_t = uint16_t;
using float32_t = float;
constexpr uint32_t recordRuns = 100u;
constexpr int ARGMAX_LABEL = 63127;

template <typename DataT>
static inline void fillIndex(DataT* mat, uint32_t m, uint32_t n)
{
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; j++)
        {
        // Force only certain index to have a value, the rest is set to 0.
        mat[i * n + j] = j == ARGMAX_LABEL ? static_cast<DataT>(250.0) : static_cast<DataT>(0.0);
        }
    }
}

__global__
void argMax(half* inputBuffer, int *outputBuffer, int reductionSize)
{
  uint laneID = threadIdx.x;
  uint laneCount = blockDim.x;
  half laneMax = inputBuffer[laneID];
  uint laneResult = laneID;

  // 64000/64 -> 1000
  uint numBatches = reductionSize / warpSize;
  for (int i = 1; i < numBatches; ++i) {
    uint idx = laneCount * i + laneID;
    half new_in = inputBuffer[idx];
    laneResult = new_in > laneMax ? idx : laneResult;
    laneMax = __hmax(new_in, laneMax);
  }

  // Final reduction with one subgroup
  half wgMax = __ockl_wfred_max_f16(laneMax);
  if (wgMax == laneMax) *outputBuffer = laneResult;
}

void benchmark_module(int reductionSize) {
    int batchSize = 1;

    // Initialize input matrices
    // TODO: Add support for parallel dimension.
    std::vector<float16_t> inputBuffer(batchSize * reductionSize);
    std::vector<int> outputBuffer(batchSize);

    fillIndex(inputBuffer.data(), batchSize, reductionSize);

    std::cout << "Initializing device data..." << std::endl;

    // Allocate and copy device memory
    half* d_input;
    int* d_output;

    const size_t bytesInput = inputBuffer.size() * sizeof(float16_t);
    const size_t bytesOutput = outputBuffer.size() * sizeof(int);

    CHECK_HIP_ERROR(hipMalloc(&d_input, bytesInput));
    CHECK_HIP_ERROR(hipMalloc(&d_output, bytesOutput));

    CHECK_HIP_ERROR(hipMemcpy(d_input, inputBuffer.data(), bytesInput, hipMemcpyHostToDevice));

    dim3 grid(1, 1, 1);
    dim3 block(64, 1, 1);

    std::cout << "Launching Argmax kernel..." << std::endl;
    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    CHECK_HIP_ERROR(hipEventRecord(startEvent));
    for (uint32_t i = 0; i < recordRuns; ++i) {
      argMax<<<grid, block>>>(d_input, d_output, reductionSize);
    }
    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    hipMemcpy(outputBuffer.data(), d_output, bytesOutput, hipMemcpyDeviceToHost);
    std::cout<<"Argmax result:"<<d_output[0]<<"\n";
    assert(d_output[0] == ARGMAX_LABEL && "Expected argmax to match label!");
    std::cout<<"argmax kernel successfully match label!\n";

    // Release device memoryv
    CHECK_HIP_ERROR(hipFree(d_input));
    CHECK_HIP_ERROR(hipFree(d_output));
    std::cout<< "Average time per run is:" << elapsedTimeMs/recordRuns <<" ms/iter\n";
    std::cout << "Finished!" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " reductionSize" << std::endl;
    return 1;
  }
  int reductionSize = atoi(argv[1]);
  benchmark_module(reductionSize);
  return 0;
}
