//===------- Offload API tests - olMemFill --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

struct olMemFillTest : OffloadQueueTest {
  void SetUp() override { RETURN_ON_FATAL_FAILURE(OffloadQueueTest::SetUp()); }

  template <typename PatternTy, PatternTy PatternVal, size_t Size,
            bool Block = false>
  void test_body() {
    ManuallyTriggeredTask Manual;

    // Block/enqueue tests ensure that the test has been enqueued to a queue
    // (rather than being done synchronously if the queue happens to be empty)
    if constexpr (Block) {
      ASSERT_SUCCESS(Manual.enqueue(Queue));
    }

    void *Alloc;
    ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

    PatternTy Pattern = PatternVal;
    ASSERT_SUCCESS(olMemFill(Queue, Alloc, sizeof(Pattern), &Pattern, Size));

    if constexpr (Block) {
      ASSERT_SUCCESS(Manual.trigger());
    }
    olSyncQueue(Queue);

    size_t N = Size / sizeof(Pattern);
    for (size_t i = 0; i < N; i++) {
      PatternTy *AllocPtr = reinterpret_cast<PatternTy *>(Alloc);
      ASSERT_EQ(AllocPtr[i], Pattern);
    }

    olMemFree(Alloc);
  }

  // Fill host-accessible memory (MANAGED/SHARED or HOST USM) with a raw byte
  // pattern and verify the bytes directly on the host. Used to drive the L0
  // plugin's non-fast-path fill fallbacks that stay on the CPU / reduce to a
  // single-byte fill.
  void fill_host_bytes(ol_alloc_type_t AllocType, const uint8_t *Pattern,
                       size_t PatternSize, size_t Size, bool Block = false) {
    ASSERT_EQ(Size % PatternSize, 0u);
    ManuallyTriggeredTask Manual;
    if (Block)
      ASSERT_SUCCESS(Manual.enqueue(Queue));

    void *Alloc;
    ASSERT_SUCCESS(olMemAlloc(Device, AllocType, Size, &Alloc));

    ASSERT_SUCCESS(olMemFill(Queue, Alloc, PatternSize, Pattern, Size));

    if (Block)
      ASSERT_SUCCESS(Manual.trigger());
    olSyncQueue(Queue);

    const uint8_t *AllocBytes = reinterpret_cast<const uint8_t *>(Alloc);
    for (size_t Off = 0; Off < Size; Off += PatternSize)
      ASSERT_EQ(memcmp(AllocBytes + Off, Pattern, PatternSize), 0);

    olMemFree(Alloc);
  }

  // Fill DEVICE USM (not host-dereferenceable) with a raw byte pattern, copy
  // it back to host via olMemcpy, and verify. Drives the L0 plugin's on-device
  // replicate fallback for mixed-byte patterns.
  void fill_device_bytes(const uint8_t *Pattern, size_t PatternSize,
                         size_t Size, bool Block = false) {
    ASSERT_EQ(Size % PatternSize, 0u);
    ManuallyTriggeredTask Manual;
    if (Block)
      ASSERT_SUCCESS(Manual.enqueue(Queue));

    void *Alloc;
    ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));

    ASSERT_SUCCESS(olMemFill(Queue, Alloc, PatternSize, Pattern, Size));

    std::vector<uint8_t> Output(Size, 0);
    ASSERT_SUCCESS(
        olMemcpy(Queue, Output.data(), Host, Alloc, Device, Size));

    if (Block)
      ASSERT_SUCCESS(Manual.trigger());
    ASSERT_SUCCESS(olSyncQueue(Queue));

    for (size_t Off = 0; Off < Size; Off += PatternSize)
      ASSERT_EQ(memcmp(Output.data() + Off, Pattern, PatternSize), 0);

    olMemFree(Alloc);
  }
};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemFillTest);

TEST_P(olMemFillTest, Success8) { test_body<uint8_t, 0x42, 1024>(); }
TEST_P(olMemFillTest, Success8NotMultiple4) {
  test_body<uint8_t, 0x42, 1023>();
}
TEST_P(olMemFillTest, Success8Enqueue) {
  test_body<uint8_t, 0x42, 1024, true>();
}
TEST_P(olMemFillTest, Success8NotMultiple4Enqueue) {
  test_body<uint8_t, 0x42, 1023, true>();
}

TEST_P(olMemFillTest, Success16) { test_body<uint8_t, 0x42, 1024>(); }
TEST_P(olMemFillTest, Success16NotMultiple4) {
  test_body<uint16_t, 0x4243, 1022>();
}
TEST_P(olMemFillTest, Success16Enqueue) {
  test_body<uint8_t, 0x42, 1024, true>();
}
TEST_P(olMemFillTest, Success16NotMultiple4Enqueue) {
  test_body<uint16_t, 0x4243, 1022, true>();
}

TEST_P(olMemFillTest, Success32) { test_body<uint32_t, 0xDEADBEEF, 1024>(); }
TEST_P(olMemFillTest, Success32Enqueue) {
  test_body<uint32_t, 0xDEADBEEF, 1024, true>();
}

TEST_P(olMemFillTest, SuccessLarge) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  struct PatternT {
    uint64_t A;
    uint64_t B;
  } Pattern{UINT64_MAX, UINT64_MAX};

  ASSERT_SUCCESS(olMemFill(Queue, Alloc, sizeof(Pattern), &Pattern, Size));

  olSyncQueue(Queue);

  size_t N = Size / sizeof(Pattern);
  for (size_t i = 0; i < N; i++) {
    PatternT *AllocPtr = reinterpret_cast<PatternT *>(Alloc);
    ASSERT_EQ(AllocPtr[i].A, UINT64_MAX);
    ASSERT_EQ(AllocPtr[i].B, UINT64_MAX);
  }

  olMemFree(Alloc);
}

TEST_P(olMemFillTest, SuccessLargeEnqueue) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ManuallyTriggeredTask Manual;
  ASSERT_SUCCESS(Manual.enqueue(Queue));

  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  struct PatternT {
    uint64_t A;
    uint64_t B;
  } Pattern{UINT64_MAX, UINT64_MAX};

  ASSERT_SUCCESS(olMemFill(Queue, Alloc, sizeof(Pattern), &Pattern, Size));

  Manual.trigger();
  olSyncQueue(Queue);

  size_t N = Size / sizeof(Pattern);
  for (size_t i = 0; i < N; i++) {
    PatternT *AllocPtr = reinterpret_cast<PatternT *>(Alloc);
    ASSERT_EQ(AllocPtr[i].A, UINT64_MAX);
    ASSERT_EQ(AllocPtr[i].B, UINT64_MAX);
  }

  olMemFree(Alloc);
}

TEST_P(olMemFillTest, SuccessLargeByteAligned) {
  constexpr size_t Size = 17 * 64;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  struct __attribute__((packed)) PatternT {
    uint64_t A;
    uint64_t B;
    uint8_t C;
  } Pattern{UINT64_MAX, UINT64_MAX, 255};

  ASSERT_SUCCESS(olMemFill(Queue, Alloc, sizeof(Pattern), &Pattern, Size));

  olSyncQueue(Queue);

  size_t N = Size / sizeof(Pattern);
  for (size_t i = 0; i < N; i++) {
    PatternT *AllocPtr = reinterpret_cast<PatternT *>(Alloc);
    ASSERT_EQ(AllocPtr[i].A, UINT64_MAX);
    ASSERT_EQ(AllocPtr[i].B, UINT64_MAX);
    ASSERT_EQ(AllocPtr[i].C, 255);
  }

  olMemFree(Alloc);
}

TEST_P(olMemFillTest, SuccessLargeByteAlignedEnqueue) {
  SKIP_KNOWN_FAILURE(LevelZero{"unsupported feature"});
  constexpr size_t Size = 17 * 64;
  void *Alloc;
  ManuallyTriggeredTask Manual;
  ASSERT_SUCCESS(Manual.enqueue(Queue));

  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  struct __attribute__((packed)) PatternT {
    uint64_t A;
    uint64_t B;
    uint8_t C;
  } Pattern{UINT64_MAX, UINT64_MAX, 255};

  ASSERT_SUCCESS(olMemFill(Queue, Alloc, sizeof(Pattern), &Pattern, Size));

  Manual.trigger();
  olSyncQueue(Queue);

  size_t N = Size / sizeof(Pattern);
  for (size_t i = 0; i < N; i++) {
    PatternT *AllocPtr = reinterpret_cast<PatternT *>(Alloc);
    ASSERT_EQ(AllocPtr[i].A, UINT64_MAX);
    ASSERT_EQ(AllocPtr[i].B, UINT64_MAX);
    ASSERT_EQ(AllocPtr[i].C, 255);
  }

  olMemFree(Alloc);
}

TEST_P(olMemFillTest, InvalidPatternSize) {
  constexpr size_t Size = 1025;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  uint16_t Pattern = 0x4242;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olMemFill(Queue, Alloc, sizeof(Pattern), &Pattern, Size));

  olSyncQueue(Queue);
  olMemFree(Alloc);
}

// The tests below drive fill patterns that cannot use a plain power-of-two
// native device fill, exercising the plugin fallbacks (e.g. the Level Zero
// plugin's memoryFillFallback). They select behavior purely through pattern
// shape (size / byte content / allocation type) and are valid on any backend.

// Non-power-of-two size with all identical bytes: falls off the native
// power-of-two fast path, but reduces to an equivalent single-byte fill.
TEST_P(olMemFillTest, SuccessRepeatedByteNotPow2) {
  const uint8_t Pattern[3] = {0xAB, 0xAB, 0xAB};
  fill_host_bytes(OL_ALLOC_TYPE_MANAGED, Pattern, sizeof(Pattern), 3 * 341);
}

TEST_P(olMemFillTest, SuccessRepeatedByteNotPow2Enqueue) {
  const uint8_t Pattern[3] = {0xAB, 0xAB, 0xAB};
  fill_host_bytes(OL_ALLOC_TYPE_MANAGED, Pattern, sizeof(Pattern), 3 * 341,
                  /*Block=*/true);
}

// Mixed-byte, non-power-of-two pattern on host-accessible memory: the fill can
// be completed on the CPU without a device fill.
TEST_P(olMemFillTest, SuccessMixedBytesManaged) {
  const uint8_t Pattern[3] = {0x01, 0x02, 0x03};
  fill_host_bytes(OL_ALLOC_TYPE_MANAGED, Pattern, sizeof(Pattern), 3 * 341);
}

TEST_P(olMemFillTest, SuccessMixedBytesManagedEnqueue) {
  const uint8_t Pattern[3] = {0x01, 0x02, 0x03};
  fill_host_bytes(OL_ALLOC_TYPE_MANAGED, Pattern, sizeof(Pattern), 3 * 341,
                  /*Block=*/true);
}

// Same pattern on explicit HOST USM to cover the HOST (not just SHARED) branch.
TEST_P(olMemFillTest, SuccessMixedBytesHost) {
  const uint8_t Pattern[3] = {0x01, 0x02, 0x03};
  fill_host_bytes(OL_ALLOC_TYPE_HOST, Pattern, sizeof(Pattern), 3 * 341);
}

// Mixed-byte, non-power-of-two pattern on DEVICE memory: the plugin must build
// the fill on the device (e.g. seed one copy then replicate it). Verified by
// copying the result back to the host.
TEST_P(olMemFillTest, SuccessMixedBytesDeviceNotPow2) {
  const uint8_t Pattern[3] = {0x01, 0x02, 0x03};
  fill_device_bytes(Pattern, sizeof(Pattern), 3 * 341);
}

TEST_P(olMemFillTest, SuccessMixedBytesDeviceNotPow2Enqueue) {
  // The device-side fallback fill synchronizes the queue internally, which
  // deadlocks against a manually blocked host function on the Level Zero
  // backend (mirrors SuccessLargeByteAlignedEnqueue).
  SKIP_KNOWN_FAILURE(LevelZero{"unsupported feature"});
  const uint8_t Pattern[3] = {0x01, 0x02, 0x03};
  fill_device_bytes(Pattern, sizeof(Pattern), 3 * 341, /*Block=*/true);
}

// Mixed-byte, oversized power-of-two pattern on DEVICE memory: larger than the
// device's maximum native fill pattern size on typical Level Zero hardware, so
// it also takes the on-device replicate path there while remaining correct
// everywhere.
TEST_P(olMemFillTest, SuccessMixedBytesDeviceOversized) {
  uint8_t Pattern[32];
  for (size_t I = 0; I < sizeof(Pattern); I++)
    Pattern[I] = static_cast<uint8_t>(I + 1);
  fill_device_bytes(Pattern, sizeof(Pattern), 32 * 32);
}

TEST_P(olMemFillTest, SuccessMixedBytesDeviceOversizedEnqueue) {
  // See SuccessMixedBytesDeviceNotPow2Enqueue: the device-side fallback fill
  // synchronizes internally and deadlocks against the blocked host function.
  SKIP_KNOWN_FAILURE(LevelZero{"unsupported feature"});
  uint8_t Pattern[32];
  for (size_t I = 0; I < sizeof(Pattern); I++)
    Pattern[I] = static_cast<uint8_t>(I + 1);
  fill_device_bytes(Pattern, sizeof(Pattern), 32 * 32, /*Block=*/true);
}
