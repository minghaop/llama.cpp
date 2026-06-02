//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <cstdlib>
#include <sstream>
#ifndef PATH_MAX
#include <climits>
#endif

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
  std::string rc     = path;
  const size_t index = path.find_last_of(pal::Path::getSeparator());
  if (index != std::string::npos) {
    rc = path.substr(0, index);
  }
  return rc;
}

#ifndef __hexagon__
//------------------------------------------------------------------------------
//    pal::Path::getAbsolute
//------------------------------------------------------------------------------
std::string pal::Path::getAbsolute(const std::string &path) {
  // Functionality was duplicated of function in FileOp
  // Just call that function directly instead
  return pal::FileOp::getAbsolutePath(path);
}
#endif

//------------------------------------------------------------------------------
//    pal::Path::isAbsolute
//------------------------------------------------------------------------------
bool pal::Path::isAbsolute(const std::string &path) {
  return path.size() > 0 && path[0] == getSeparator();
}
