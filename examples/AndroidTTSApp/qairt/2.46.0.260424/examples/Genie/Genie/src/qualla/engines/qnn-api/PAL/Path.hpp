//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

//------------------------------------------------------------------------------
/// @file
///   The file includes APIs for path related operations on supported platforms
//------------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

namespace pal {

struct Path {
  //---------------------------------------------------------------------------
  /// Path constants
  //---------------------------------------------------------------------------
  static const std::string CURRENT_DIR_TOKEN;
  static const std::string PARENT_DIR_TOKEN;

  //----------------------------------------------------------------------------
  //    @brief Returns path separator for the system
  //------------------------------------------------------------------------------
  static inline char getSeparator() { return '/'; }

  //---------------------------------------------------------------------------
  /// @brief Concatenate multiple strings into a path
  //---------------------------------------------------------------------------
  static inline std::string combine(const std::string &s1) {
    return s1;
  }

  // Operates like python os.path.join: pal::Path::combine(path1, path2, path3, ...)
  template<typename... Parts>
  static std::string combine(const std::string &s1, Parts... sN) {
    return s1 + (!s1.empty() && s1.back() != '/' && s1.back() != '\\' ? std::string(1, getSeparator()) : "") + combine(sN...);
  }

  //---------------------------------------------------------------------------
  /// @brief Split a path into parts divided by the path seperator
  //---------------------------------------------------------------------------
  static std::vector<std::string> split(const std::string &path);

  //---------------------------------------------------------------------------
  /// @brief Get the file name
  //---------------------------------------------------------------------------
  static std::string getFileName(const std::string &path);

  //---------------------------------------------------------------------------
  /// @brief Get the directory name
  //---------------------------------------------------------------------------
  static std::string getDirectoryName(const std::string &path);

  //---------------------------------------------------------------------------
  /// @brief Get absolute path
  //---------------------------------------------------------------------------
  static std::string getAbsolute(const std::string &path);

  //---------------------------------------------------------------------------
  /// @brief Check if the input path is absolute path
  //---------------------------------------------------------------------------
  static bool isAbsolute(const std::string &path);
};

} // ns pal
