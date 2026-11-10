//===------- Offload API tests - olMemAllocHost with several devices ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

#include <cstring>
#include <vector>

namespace {
ol_platform_handle_t getPlatform(ol_device_handle_t Device) {
  ol_platform_handle_t Platform = nullptr;
  olGetDeviceInfo(Device, OL_DEVICE_INFO_PLATFORM, sizeof(Platform), &Platform);
  return Platform;
}

uint32_t getDriverId(ol_device_handle_t Device) {
  uint32_t DriverId = 0;
  olGetDeviceInfo(Device, OL_DEVICE_INFO_DRIVER_ID, sizeof(DriverId),
                  &DriverId);
  return DriverId;
}
} // namespace

// Host allocations are not bound to a device: one made in a context with
// several devices must be usable from every device of that context, not only
// from the device whose allocator backs the context's host pool.
struct olMemAllocHostMultiDeviceTest : OffloadDeviceTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadDeviceTest::SetUp());

    // Put the parameter device last, so that it is not the device backing the
    // context's host allocations.
    for (const auto &D : TestEnvironment::getDevices())
      if (D.Handle != Device && getPlatform(D.Handle) == getPlatform(Device) &&
          getDriverId(D.Handle) == getDriverId(Device))
        Devices.push_back(D.Handle);
    Devices.push_back(Device);
    if (Devices.size() < 2)
      GTEST_SKIP() << "Needs at least two devices sharing a platform and a "
                      "driver.";

    ASSERT_SUCCESS(
        olCreateContext(Devices.size(), Devices.data(), &MultiContext));
    for (ol_device_handle_t D : Devices) {
      ol_queue_handle_t Queue = nullptr;
      ASSERT_SUCCESS(olCreateQueue(MultiContext, D, &Queue));
      Queues.push_back(Queue);
    }
  }

  void TearDown() override {
    for (ol_queue_handle_t Queue : Queues)
      olDestroyQueue(Queue);
    if (MultiContext)
      olDestroyContext(MultiContext);
    RETURN_ON_FATAL_FAILURE(OffloadDeviceTest::TearDown());
  }

  std::vector<ol_device_handle_t> Devices;
  std::vector<ol_queue_handle_t> Queues;
  ol_context_handle_t MultiContext = nullptr;
};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemAllocHostMultiDeviceTest);

TEST_P(olMemAllocHostMultiDeviceTest, SuccessCopyOnEveryDevice) {
  // Include a size above the AMDGPU async copy limit (1 MiB by default), which
  // takes a slower path when the host buffer is not known to be pinned.
  for (size_t Size : {size_t(64), size_t(16 * 1024), size_t(2 * 1024 * 1024)}) {
    void *Input = nullptr;
    void *Output = nullptr;
    ASSERT_SUCCESS(olMemAllocHost(MultiContext, Size, &Input));
    ASSERT_SUCCESS(olMemAllocHost(MultiContext, Size, &Output));
    for (size_t I = 0; I < Size; ++I)
      static_cast<uint8_t *>(Input)[I] = static_cast<uint8_t>(I % 251);

    for (size_t Idx = 0; Idx < Devices.size(); ++Idx) {
      ol_device_handle_t D = Devices[Idx];
      void *DevAlloc = nullptr;
      ASSERT_SUCCESS(
          olMemAlloc(MultiContext, D, OL_ALLOC_TYPE_DEVICE, Size, &DevAlloc));

      // Asynchronous copies on the device's queue.
      std::memset(Output, 0, Size);
      ASSERT_SUCCESS(olMemcpy(Queues[Idx], DevAlloc, D, Input, Host, Size));
      ASSERT_SUCCESS(olMemcpy(Queues[Idx], Output, Host, DevAlloc, D, Size));
      ASSERT_SUCCESS(olSyncQueue(Queues[Idx]));
      ASSERT_EQ(std::memcmp(Input, Output, Size), 0)
          << "queued copy, device " << Idx << ", size " << Size;

      // Synchronous copies.
      std::memset(Output, 0, Size);
      ASSERT_SUCCESS(olMemcpy(nullptr, DevAlloc, D, Input, Host, Size));
      ASSERT_SUCCESS(olMemcpy(nullptr, Output, Host, DevAlloc, D, Size));
      ASSERT_EQ(std::memcmp(Input, Output, Size), 0)
          << "synchronous copy, device " << Idx << ", size " << Size;

      ASSERT_SUCCESS(olMemFree(MultiContext, DevAlloc));
    }

    ASSERT_SUCCESS(olMemFree(MultiContext, Input));
    ASSERT_SUCCESS(olMemFree(MultiContext, Output));
  }
}

TEST_P(olMemAllocHostMultiDeviceTest, SuccessLaunchOnEveryDevice) {
  ol_kernel_launch_size_args_t LaunchArgs{};
  LaunchArgs.Dimensions = 1;
  LaunchArgs.NumGroups = {1, 1, 1};
  LaunchArgs.GroupSize = {64, 1, 1};
  const size_t SliceSize = LaunchArgs.GroupSize.x;

  void *Mem = nullptr;
  ASSERT_SUCCESS(olMemAllocHost(
      MultiContext, Devices.size() * SliceSize * sizeof(uint32_t), &Mem));
  std::memset(Mem, 0xff, Devices.size() * SliceSize * sizeof(uint32_t));

  std::vector<std::unique_ptr<llvm::MemoryBuffer>> Binaries(Devices.size());
  std::vector<ol_program_handle_t> Programs(Devices.size(), nullptr);
  for (size_t Idx = 0; Idx < Devices.size(); ++Idx) {
    ol_device_handle_t D = Devices[Idx];
    ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", D, Binaries[Idx]));
    ASSERT_SUCCESS(
        olCreateProgram(MultiContext, D, Binaries[Idx]->getBufferStart(),
                        Binaries[Idx]->getBufferSize(), &Programs[Idx]));
    ol_symbol_handle_t Kernel = nullptr;
    ASSERT_SUCCESS(
        olGetSymbol(Programs[Idx], "foo", OL_SYMBOL_KIND_KERNEL, &Kernel));

    // Each device writes its own slice of the shared host buffer.
    uint32_t *Slice = static_cast<uint32_t *>(Mem) + Idx * SliceSize;
    void *ArgPtrs[] = {&Slice};
    size_t ArgSizes[] = {sizeof(Slice)};
    ASSERT_SUCCESS(olLaunchKernel(Queues[Idx], D, Kernel, &LaunchArgs, nullptr,
                                  std::size(ArgPtrs), ArgPtrs, ArgSizes));
  }

  for (ol_queue_handle_t Queue : Queues)
    ASSERT_SUCCESS(olSyncQueue(Queue));

  uint32_t *Data = static_cast<uint32_t *>(Mem);
  for (size_t Idx = 0; Idx < Devices.size(); ++Idx)
    for (uint32_t I = 0; I < SliceSize; ++I)
      ASSERT_EQ(Data[Idx * SliceSize + I], I) << "device " << Idx;

  for (ol_program_handle_t Program : Programs)
    ASSERT_SUCCESS(olDestroyProgram(Program));
  ASSERT_SUCCESS(olMemFree(MultiContext, Mem));
}

TEST_P(olMemAllocHostMultiDeviceTest, SuccessInfoOnEveryDevice) {
  constexpr size_t Size = 4096;
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAllocHost(MultiContext, Size, &Ptr));

  ol_alloc_type_t Type;
  ASSERT_SUCCESS(
      olGetMemInfo(MultiContext, Ptr, OL_MEM_INFO_TYPE, sizeof(Type), &Type));
  ASSERT_EQ(Type, OL_ALLOC_TYPE_HOST);

  void *Base = nullptr;
  ASSERT_SUCCESS(
      olGetMemInfo(MultiContext, Ptr, OL_MEM_INFO_BASE, sizeof(Base), &Base));
  ASSERT_EQ(Base, Ptr);

  size_t AllocSize = 0;
  ASSERT_SUCCESS(olGetMemInfo(MultiContext, Ptr, OL_MEM_INFO_SIZE,
                              sizeof(AllocSize), &AllocSize));
  ASSERT_GE(AllocSize, Size);

  ol_device_handle_t Owner = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_ARGUMENT,
               olGetMemInfo(MultiContext, Ptr, OL_MEM_INFO_DEVICE,
                            sizeof(Owner), &Owner));

  for (ol_device_handle_t D : Devices) {
    bool IsAccessible = false;
    ASSERT_SUCCESS(
        olMemIsAccessible(MultiContext, D, Ptr, Size, &IsAccessible));
    ASSERT_TRUE(IsAccessible);
  }

  ASSERT_SUCCESS(olMemFree(MultiContext, Ptr));
}

TEST_P(olMemAllocHostMultiDeviceTest, SuccessRepeatedAllocFree) {
  // Above the host pool size threshold (8 KiB), so every allocation reaches
  // the backend and is registered and unregistered as pinned memory. Backends
  // tend to reuse the address, which exposes unbalanced bookkeeping.
  constexpr size_t Size = 64 * 1024;
  for (int I = 0; I < 8; ++I) {
    void *Ptr = nullptr;
    ASSERT_SUCCESS(olMemAllocHost(MultiContext, Size, &Ptr));
    ASSERT_SUCCESS(olMemFree(MultiContext, Ptr));
  }
}

TEST_P(olMemAllocHostMultiDeviceTest, SuccessRegisterInteriorPointer) {
  constexpr size_t Size = 64 * 1024;
  constexpr size_t Offset = 64;
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAllocHost(MultiContext, Size, &Ptr));
  void *Interior = static_cast<char *>(Ptr) + Offset;

  for (ol_device_handle_t D : Devices) {
    void *Locked = nullptr;
    ASSERT_SUCCESS(olMemRegister(D, Interior, Offset,
                                 OL_MEMORY_REGISTER_FLAG_LOCK_MEMORY, &Locked));
    // Host allocations are accessible through their host address, so the
    // device accessible pointer must keep the offset into the allocation.
    ASSERT_EQ(Locked, Interior);
    ASSERT_SUCCESS(
        olMemUnregister(D, Interior, OL_MEMORY_REGISTER_FLAG_UNLOCK_MEMORY));
  }

  ASSERT_SUCCESS(olMemFree(MultiContext, Ptr));
}
