//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <shlwapi.h>
#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <sstream>

#include "Common.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"


const std::string pal::Path::CURRENT_DIR_TOKEN = ".";
const std::string pal::Path::PARENT_DIR_TOKEN = "..";

//------------------------------------------------------------------------------
//    pal::Path::split
//------------------------------------------------------------------------------
std::vector<std::string> pal::Path::split(const std::string &path) {
  std::vector<std::string> components;

  std::string token;
  std::istringstream tokenStream(path);
  while(std::getline(tokenStream, token, getSeparator())) {
    components.push_back(token);
  }

  return components;
}

//------------------------------------------------------------------------------
//    pal::Path::getFileName
//------------------------------------------------------------------------------
std::string pal::Path::getFileName(const std::string &path) {
  return path.substr(getDirectoryName(path).size() + 1);
}

//------------------------------------------------------------------------------
//    pal::Path::getDirectoryName
//------------------------------------------------------------------------------
std::string pal::Path::getDirectoryName(const std::string &path) {
  std::string rc = path;
  int32_t index  = std::max(static_cast<int32_t>(path.find_last_of('\\')),
                            static_cast<int32_t>(path.find_last_of('/')));
  if (index != static_cast<int32_t>(std::string::npos)) {
    rc = path.substr(0, index);
  }
  pal::normalizeSeparator(rc);
  return rc;
}

//------------------------------------------------------------------------------
//    pal::Path::getAbsolute
//------------------------------------------------------------------------------
std::string pal::Path::getAbsolute(const std::string &path) {
  std::string res = pal::FileOp::getAbsolutePath(path);
  pal::normalizeSeparator(res);
  return res;
}

//------------------------------------------------------------------------------
//    pal::Path::isAbsolute
//    requirement : shlwapi.lib
//------------------------------------------------------------------------------
bool pal::Path::isAbsolute(const std::string &path) {
  std::string windowsPath = path;
  // in windows, when we need to check relative or absolute path,
  // separator MUST be '\\' rather than '/'
  // for more information : https://docs.microsoft.com/en-us/dotnet/standard/io/file-path-formats
  replace(windowsPath.begin(), windowsPath.end(), '/', '\\');
  return PathIsRelativeA(windowsPath.c_str()) == false;
}
