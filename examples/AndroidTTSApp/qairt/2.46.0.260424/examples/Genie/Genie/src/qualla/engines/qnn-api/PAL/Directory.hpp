//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

//---------------------------------------------------------------------------
/// @file
///   This file includes APIs for directory operations on supported platforms
//---------------------------------------------------------------------------

#pragma once

#include <string>

#include "PAL/FileOp.hpp"

namespace pal {
class Directory;
}

class pal::Directory {
 public:
  using DirMode = pal::FileOp::FileMode;
  //---------------------------------------------------------------------------
  /// @brief
  ///   Creates a directory in the file system.
  /// @param path
  ///   Name of directory to create.
  /// @param dirmode
  ///   Directory mode
  /// @return
  ///   True if
  ///     1. create a directory successfully
  ///     2. or directory exist already
  ///   False otherwise
  ///
  ///  For example:
  ///
  ///  - Create a directory in default.
  ///     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ///     pal::Directory::Create(path, pal::Directory::DirMode::S_DEFAULT_);
  ///     pal::Directory::Create(path);
  ///     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ///
  ///  - Create a directory with specific permission.
  ///     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ///     pal::Directory::Create(path, pal::Directory::DirMode::S_IRWXU_|
  ///                                  pal::Directory::DirMode::S_IRWXG_|
  ///                                  pal::Directory::DirMode::S_IRWXO_);
  ///     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ///
  /// @note For windows, dirmode is not used.
  /// @note For linux, dirmode is used to set the permission of the folder.
  //---------------------------------------------------------------------------
  static bool create(const std::string &path,
                     pal::Directory::DirMode dirmode = pal::Directory::DirMode::S_DEFAULT_);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Removes the entire directory whether it's empty or not.
  /// @param path
  ///   Name of directory to delete.
  /// @return
  ///   True if the directory was successfully deleted, false otherwise.
  //---------------------------------------------------------------------------
  static bool remove(const std::string &path);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Creates a directory and all parent directories required.
  /// @param path
  ///   Path of directory to create.
  /// @return
  ///   True if the directory was successfully created, false otherwise.
  //---------------------------------------------------------------------------
  static bool makePath(const std::string &path);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Returns the full path to the temporary directory.
  /// @return
  ///   If the temp directory is created sucessfully, returns the path to the
  ///   newly created folder. Otherwise, the empty string will be returned.
  /// @note
  ///   + On Linux, the temporary directory is under "/tmp".
  ///      For example: "/tmp/tmp.ttAvZ2"
  ///   + On Windows, the temporary directory is under
  ///     "C:\Users\[UserName]\AppData\Local\Temp". For example:
  ///     "C:\Users\JOHN\AppData\Local\Temp\tmpa00980"
  ///   + The API creates a new temporary folder for each call.
  //---------------------------------------------------------------------------
  static std::string getTempDirectory();

  //---------------------------------------------------------------------------
  /// @brief
  ///   Checks if the input directory path has write permissions
  /// @param path
  ///   Path of directory to be checked.
  /// @return
  ///   True if the directory is writable, false otherwise.
  //---------------------------------------------------------------------------
  static bool isWritable(const std::string &path);
};
