//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "fmt/format.h"
#include "qualla/detail/gpio-marker.hpp"

namespace fs = std::filesystem;

namespace qualla {

GpioMarker::GpioMarker(const nlohmann::json& conf) {
  // Parse config
  using qc = qualla::Config;

  _tool_path = qc::optional<std::string>(conf, "tool-path", "");
  _command   = qc::optional<std::string>(conf, "command", "");
  _gpio_num  = qc::optional<int32_t>(conf, "gpio-num", -1);

  if (!_tool_path.empty()) {
    if (fs::exists(_tool_path)) {
      _gpio_marker_enable = true;
      reset();
    } else {
      _gpio_marker_enable = false;
    }
  } else {
    _gpio_marker_enable = false;
  }
}

GpioMarker::~GpioMarker() {}

void GpioMarker::set() {
  if (!_gpio_marker_enable) return;

  _gpio_status    = !_gpio_status;
  std::string cmd = fmt::format("{} {} {}={}", _tool_path, _command, _gpio_num, _gpio_status);
  int unused = system(cmd.c_str()); //unused variable
  (void)unused;
}

void GpioMarker::reset() {
  if (!_gpio_marker_enable) return;

  std::string cmd = fmt::format("{} {} {}=0", _tool_path, _command, _gpio_num);
  int unused = system(cmd.c_str()); //unused variable
  (void)unused;
  _gpio_status = 0;
}

std::unique_ptr<GpioMarker> GpioMarker::create(const nlohmann::json& conf) {
  return std::make_unique<GpioMarker>(conf);
}

std::unique_ptr<GpioMarker> GpioMarker::create(std::istream& json_stream) {
  return create(nlohmann::json::parse(json_stream));
}

std::unique_ptr<GpioMarker> GpioMarker::create(const std::string& json_str) {
  return create(nlohmann::json::parse(json_str));
}

}  // namespace qualla
