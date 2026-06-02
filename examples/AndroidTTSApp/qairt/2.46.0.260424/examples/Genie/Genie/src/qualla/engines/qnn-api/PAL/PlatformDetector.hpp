//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

//---------------------------------------------------------------------------
/// @file
///   This file includes APIs for checking if library versioning is supported.
/// @note
///   Currently versioning is enabled for Ubuntu platform.
//---------------------------------------------------------------------------

#pragma once

#include <string>

#define CDSP_VERSION_MAJOR 1

namespace pal {

/**
 * @brief Utility class for runtime platform detection to determine library
 * versioning is enabled
 *
 * This class provides runtime detection of the target platform to determine
 * whether to use versioned (.so.X) or unversioned (.so) library names.
 *
 * Detection Logic:
 * - Ubuntu platforms: Use versioned libraries with different version suffixes
 * based on library name:
 *   * SNPE/QNN libraries: Use AISW_VERSION_MAJOR (e.g., libQnnHtp.so.2,
 * libSNPE.so.2)
 *   * CDSP/ADSP libraries: Use CDSP_VERSION_MAJOR (e.g., libcdsprpc.so.1,
 * libadsprpc.so.1)
 *   * Other libraries: No version suffix (e.g., libion.so, libdmabufheap.so)
 * - OE-Linux platforms: Use unversioned libraries for all (e.g., libcdsprpc.so,
 * libSNPE.so)
 */
class PlatformDetector {
public:
  /**
   * @brief Check if current platform supports library versioning
   * @return true for Ubuntu, false for OE-Linux and other platforms
   */
  static bool isPlatformUbuntu();

  /**
   * @brief Update the filename for versioned libraries
   * @return Append version based on soname
   */
  static const char *updateLibraryWithVersionSuffix(const char *filename);
};

} // namespace pal