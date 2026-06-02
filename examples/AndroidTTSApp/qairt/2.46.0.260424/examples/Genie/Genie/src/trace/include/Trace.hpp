//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <cstdint>
#include <cstdlib>

namespace genie {
namespace profiling {

class Traceable;

/**
 * The FunctionTracer class is at the heart of the GENIE_TRACE macro
 * The macro initializes a FunctionTracer object functionTracer__ with the function name
 * The destructor is implicity called the end of the scope/function, and updates the logbook
 */
class FunctionTracer final {
 private:
  const Traceable& m_traceObject;
  const char* m_functionName;
  const uint64_t m_startTime{0ul};
  const size_t m_depth{0u};

 public:
  /**
   * Records the time of construction.
   *
   * @param traceObject the object whose function is being traced. If traceObject's trace logger is
   * null, then this trace will be ignored.
   * @param name        the name of the trace event.
   */
  FunctionTracer(const Traceable& traceObject, const char* name);
  ~FunctionTracer();

  FunctionTracer()                                 = delete;
  FunctionTracer(const FunctionTracer&)            = delete;
  FunctionTracer(FunctionTracer&&)                 = delete;
  FunctionTracer& operator=(const FunctionTracer&) = delete;
  FunctionTracer& operator=(FunctionTracer&&)      = delete;

  uint64_t getStartTimeInUs() const { return m_startTime; }
};

}  // namespace profiling
}  // namespace genie
