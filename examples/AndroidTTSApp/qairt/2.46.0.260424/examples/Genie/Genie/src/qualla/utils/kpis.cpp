//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>

#include "qualla/detail/kpi.hpp"

namespace qualla {

std::string Kpi::dump(std::string_view sep) const {
  return fmt::format(
      "last:{:.2f}{}total:{:.2f}{}min:{:.2f}{}max:{:.2f}{}avg:{:.2f} (msec){}count:{}",
      static_cast<double>(last_usec) / 1000.0,
      sep,
      static_cast<double>(total_usec) / 1000.0,
      sep,
      static_cast<double>(min_usec) / 1000.0,
      sep,
      static_cast<double>(max_usec) / 1000.0,
      sep,
      static_cast<double>(total_usec) / static_cast<double>(count ? count : 1) / 1000.0,
      sep,
      count);
}

}  // namespace qualla
