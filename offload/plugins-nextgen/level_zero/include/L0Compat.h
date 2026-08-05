//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Level Zero compatibility layer enabling us to compile using new APIs.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0COMPAT_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0COMPAT_H

#include "APIHelpers.h"

#include <level_zero/ze_api.h>

#include <utility>

namespace llvm::omp::target::plugin::detail {

#ifdef _WIN32
using ZeHostFunctionCallback = void(__stdcall *)(void *);
#else
using ZeHostFunctionCallback = void (*)(void *);
#endif

} // namespace llvm::omp::target::plugin::detail

API_HELPER_OPTIONAL(ze_result_t, zeCommandListAppendLaunchKernelWithArguments,
                    ze_command_list_handle_t hCommandList,
                    ze_kernel_handle_t hKernel,
                    const ze_group_count_t groupCounts,
                    const ze_group_size_t groupSizes, void **pArguments,
                    const void *pNext, ze_event_handle_t hSignalEvent,
                    uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);

API_HELPER_OPTIONAL(ze_context_handle_t, zeDriverGetDefaultContext,
                    ze_driver_handle_t hDriver);

API_HELPER_OPTIONAL(
    ze_result_t, zeCommandListAppendHostFunction,
    ze_command_list_handle_t hCommandList,
    llvm::omp::target::plugin::detail::ZeHostFunctionCallback pfnHostFunction,
    void *pUserData, const void *pNext, ze_event_handle_t hSignalEvent,
    uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);

namespace llvm::omp::target::plugin::detail {

template <auto Fn> struct TryZeMaybeUnsupported {
  template <typename Callable> ze_result_t operator+(Callable &&Call) const {
    auto &HasReturnedUnsupported = api_helper::hasReturnedUnsupported<Fn>;
    if (!api_helper::canCall<Fn>() ||
        HasReturnedUnsupported.load(std::memory_order_relaxed))
      return ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;

    ze_result_t Result = std::forward<Callable>(Call)();
    if (Result == ZE_RESULT_ERROR_UNSUPPORTED_FEATURE)
      HasReturnedUnsupported.store(true, std::memory_order_relaxed);
    return Result;
  }
};

} // namespace llvm::omp::target::plugin::detail

#define TRY_ZE_MAYBE_UNSUPPORTED(Fn)                                           \
  ::llvm::omp::target::plugin::detail::TryZeMaybeUnsupported<Fn>{} + [&]()

#define CALL_ZE_RETURN_RESULT(Fn, ...)                                         \
  do {                                                                         \
    return Fn(__VA_ARGS__);                                                    \
  } while (false)

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0COMPAT_H
