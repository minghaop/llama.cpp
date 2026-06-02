//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <mutex>
#include <vector>

#include "nlohmann/json.hpp"

namespace genie {
namespace profiling {

/**
 * TraceData represents the actual data that gets stored in the logbook
 */
struct TraceData {
  const char* traceNamespace{nullptr};
  const char* functionName;
  uint64_t startTime{0ul};
  uint64_t duration{0ul};
  size_t stackDepth{0u};
};

/**
 * TraceLogger will capture trace events for Traceable objects.
 */
class TraceLogger final {
 private:
  std::mutex m_mutex;
  std::vector<TraceData> m_log;
  std::vector<std::shared_ptr<TraceLogger>> m_subLoggers;
  size_t m_id{0};

  void setTid(const size_t tid) { m_id = tid; }

 public:
  TraceLogger()  = default;
  ~TraceLogger() = default;

  TraceLogger(const TraceLogger&)            = delete;
  TraceLogger(TraceLogger&&)                 = delete;
  TraceLogger& operator=(const TraceLogger&) = delete;
  TraceLogger& operator=(TraceLogger&&)      = delete;

  /**
   * Inserts the provided event into this log.
   */
  void insert(const TraceData& event);

  /**
   * Appends serialized trace events to the provided array-like json.
   *
   * Sub-loggers will also be serialized.
   */
  void serialize(nlohmann::json& json);

  /**
   * Creates a new TraceLogger that is owned by the current logger.
   *
   * @return  a non-owning reference to the newly created logger
   */
  std::weak_ptr<TraceLogger> createSubLogger();
};

}  // namespace profiling
}  // namespace genie