//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <memory>

/**
 * Creates an RAII tracer object to profile the active scope.
 *
 * This macro is designed for classes that inherit from Traceable.
 */
#define GENIE_TRACE(...) \
  genie::profiling::FunctionTracer functionTracer__(*static_cast<const Traceable*>(this), __func__);

namespace genie {
namespace profiling {

class TraceLogger;

/**
 * A lightweight class for supporting the optional collection of trace profiling events.
 */
class Traceable {
 protected:
  std::shared_ptr<TraceLogger> m_traceLogger;

 public:
  Traceable(std::shared_ptr<TraceLogger> traceLogger) : m_traceLogger(traceLogger) {}
  virtual ~Traceable() {}

  /**
   * Sets the trace logger.
   *
   * The logger can be set to nullptr to disable trace collection.
   *
   * If a Traceable owns other Traceable objects, it should override this function
   * to propagate the logger to all Traceable children.
   */
  virtual void setTraceLogger(std::shared_ptr<TraceLogger> logger) { m_traceLogger = logger; }

  /**
   * The namespace is prepended to any trace events created by this class.
   */
  virtual const char* getTraceNamespace() const { return nullptr; }

  std::shared_ptr<TraceLogger> getTraceLogger() { return m_traceLogger; }

  friend class FunctionTracer;
};

}  // namespace profiling
}  // namespace genie