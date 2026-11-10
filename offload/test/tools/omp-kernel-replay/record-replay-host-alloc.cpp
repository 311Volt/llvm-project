// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: rm -rf %t.testdir
// RUN: mkdir -p %t.testdir
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.testdir %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: ls -t %t.testdir/*.json | sed -n '1p' | grep . | xargs -I {} %omp-kernel-replay --verify {}
// clang-format on

// Host allocations must not be served from the record-replay device memory:
// that memory is not host accessible on discrete GPUs.

// REQUIRES: gpu

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: intelgpu

#include <cstdint>
#include <cstdio>
#include <omp.h>

extern "C" void *llvm_omp_target_alloc_host(size_t Size, int DeviceNum);
extern "C" void llvm_omp_target_free_host(void *Ptr, int DeviceNum);

int main() {
  int Device = omp_get_default_device();
  size_t Size = 1000;

  // The record-replay allocator hands out increasing addresses, so a host
  // allocation taken from it would land between the two device allocations.
  char *D1 = static_cast<char *>(omp_target_alloc(64, Device));
  uint64_t *Data = static_cast<uint64_t *>(
      llvm_omp_target_alloc_host(Size * sizeof(uint64_t), Device));
  char *D2 = static_cast<char *>(omp_target_alloc(64, Device));
  char *H = reinterpret_cast<char *>(Data);
  if (!D1 || !Data || !D2 || (D1 <= H && H < D2)) {
    printf("FAIL: host allocation taken from record-replay memory\n");
    return 1;
  }

  // Crashes if the host allocation is not host accessible.
  for (size_t I = 0; I < Size; ++I)
    Data[I] = 20;

#pragma omp target teams distribute parallel for num_teams(256)                \
    thread_limit(128) map(tofrom : Data[0 : Size])
  for (size_t I = 0; I < Size; ++I)
    Data[I] = 10 + (uint64_t)I;

  uint64_t Sum = 0;
  for (size_t I = 0; I < Size; ++I)
    Sum += Data[I];

  // CHECK: PASS
  if (Sum == 509500)
    printf("PASS\n");

  llvm_omp_target_free_host(Data, Device);
  omp_target_free(D2, Device);
  omp_target_free(D1, Device);
}
