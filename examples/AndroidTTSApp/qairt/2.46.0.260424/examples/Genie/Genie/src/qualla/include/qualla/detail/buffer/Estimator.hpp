//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <string>
#include <unordered_map>

class Estimator {
 public:
  // Default constructor
  Estimator(
      std::unordered_map<uint32_t, std::unordered_map<std::string, size_t>>& contextAllocationMap) {
    // Constructor implementation
    m_contextAllocMap = contextAllocationMap;
  }

  // Destructor
  ~Estimator() {
    // Destructor implementation
    for (auto& x : m_contextAllocMap) {
      if (!x.second.empty()) {
        x.second.clear();
      }
    }
    m_contextAllocMap.clear();
  }

  // Delete copy constructor
  Estimator(const Estimator&) = delete;

  // Delete copy assignment operator
  Estimator& operator=(const Estimator&) = delete;

  // Delete move constructor
  Estimator(Estimator&&) = delete;

  // Delete move assignment operator
  Estimator& operator=(Estimator&&) = delete;

  std::unordered_map<uint32_t, std::unordered_map<std::string, size_t>>& getEstimations() {
    return m_contextAllocMap;
  }

 private:
  // {Translated ContextId -> {Tensor_name -> Size}}
  // This object is for all other backends, they must follow this method of allocation data
  std::unordered_map<uint32_t, std::unordered_map<std::string, size_t>> m_contextAllocMap;
};
