//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "TraceLogger.hpp"

namespace genie {
namespace profiling {

void TraceLogger::serialize(nlohmann::json& json) {
  std::lock_guard<std::mutex> lock(m_mutex);
  for (auto& event : m_log) {
    std::string traceName;
    if (event.traceNamespace) {
      traceName = std::string(event.traceNamespace) + "::" + event.functionName;
    } else {
      traceName = std::string(event.functionName);
    }
    json.push_back({{"name", traceName},
                    {"cat", "function"},
                    {"ph", "X"},
                    {"ts", event.startTime},
                    {"dur", event.duration},
                    {"pid", 0},
                    {"tid", m_id},
                    {"args", {{"stackDepth", event.stackDepth}}}});
  }
  for (auto& subLogger : m_subLoggers) {
    subLogger->serialize(json);
  }

  // Placeholder metadata events. Visualizer checks for these.
  //
  // Since multiple TraceLoggers may exist, these metadata events
  // should be added after the events from all TraceLoggers are
  // combined. Leaving them here for future reference.
  //
  // json.push_back({{"name", "process_name"},
  //                  {"cat", "__metadata"},
  //                  {"ph", "M"},
  //                  {"pid", 0},
  //                  {"args", {{"name", "Process 0"}}}});
  // json.push_back({{"name", "process_sort_index"},
  //                  {"cat", "__metadata"},
  //                  {"ph", "M"},
  //                  {"pid", 0},
  //                  {"args", {{"sort_index", 0}}}});
}

void TraceLogger::insert(const TraceData& event) {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_log.push_back(std::move(event));
}

std::weak_ptr<TraceLogger> TraceLogger::createSubLogger() {
  m_subLoggers.emplace_back(std::make_shared<TraceLogger>());
  m_subLoggers.back()->setTid(m_id + m_subLoggers.size());
  return m_subLoggers.back();
}

}  // namespace profiling
}  // namespace genie