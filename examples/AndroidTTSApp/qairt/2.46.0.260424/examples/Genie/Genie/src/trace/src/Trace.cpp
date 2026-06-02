//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <memory>

#include "Trace.hpp"
#include "TraceLogger.hpp"
#include "Traceable.hpp"
#include "qualla/detail/timer.hpp"

namespace genie {
namespace profiling {

thread_local size_t g_threadStackDepth{0};  // Function stack depth for each thread

static qualla::Timer<std::chrono::steady_clock, std::chrono::time_point<std::chrono::steady_clock>>&
getTraceTimer() {
  static qualla::Timer s_timer{};
  return s_timer;
}

FunctionTracer::FunctionTracer(const Traceable& traceObject, const char* name)
    : m_traceObject(traceObject),
      m_functionName(name),
      m_startTime(getTraceTimer().elapsed_usec()),
      m_depth(g_threadStackDepth++) {}

FunctionTracer::~FunctionTracer() {
  if (m_traceObject.m_traceLogger) {
    auto end_time = getTraceTimer().elapsed_usec();
    m_traceObject.m_traceLogger->insert({m_traceObject.getTraceNamespace(),
                                         m_functionName,
                                         m_startTime,
                                         end_time - m_startTime,
                                         m_depth});
  }
  --g_threadStackDepth;
}

}  // namespace profiling
}  // namespace genie
