//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "PAL/PlatformDetector.hpp"
#include "version_config.hpp"

namespace pal {

bool PlatformDetector::isPlatformUbuntu() {
  // For non-Linux platforms (Android, Windows, QNX, etc.),
  // always return empty suffix (unversioned)
  return false;
}

const char *PlatformDetector::updateLibraryWithVersionSuffix(const char *filename) {
  return filename;
}

} // namespace pal
