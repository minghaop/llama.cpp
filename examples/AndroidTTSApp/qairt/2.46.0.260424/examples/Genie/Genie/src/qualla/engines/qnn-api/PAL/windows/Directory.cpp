//=====================================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=====================================================================================

#include <Windows.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <algorithm>
#include <iostream>

#include "Common.hpp"
#include "PAL/Debug.hpp"
#include "PAL/Directory.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"

//--------------------------------------------------------------------------------------
//   pal::Directory::Create
//--------------------------------------------------------------------------------------
bool pal::Directory::create(const std::string &path, pal::Directory::DirMode dirmode) {
  struct stat st;
  // it create a directory successfully or directory exists already, return true.
  if ((stat(path.c_str(), &st) != 0 && (CreateDirectoryA(path.c_str(), NULL) != 0)) ||
      ((st.st_mode & S_IFDIR) != 0)) {
    return true;
  } else {
    DEBUG_MSG("Create Folder fail! Error code : %d", GetLastError());
  }
  return false;
}

//--------------------------------------------------------------------------------------
//   pal::Directory::Remove
//--------------------------------------------------------------------------------------
bool pal::Directory::remove(const std::string &dirName) {
  struct stat st;
  if (stat(dirName.c_str(), &st) == 0) {
    if ((st.st_mode & S_IFDIR) != 0) {
      // a directory exist and remove it !
      std::string fullPath = dirName;
      if (pal::Path::isAbsolute(dirName) == 0) {
        fullPath = pal::Path::getAbsolute(dirName);
      }
      // Note  This string MUST be double-null terminated.
      fullPath               = fullPath + '\0' + '\0';
      SHFILEOPSTRUCTA fileOp = {
          NULL,              // hwnd
          FO_DELETE,         // wFunc, delete usage
          fullPath.c_str(),  // pFrom, delete target folder
          "",                // pTo, delete operation can ignore this
          FOF_NO_UI,         // Perform operation silently, presenting no UI to user
          false,             // fAnyOperationsAborted,
          0,                 // hNameMappings
          ""                 // lpszProgressTitle, used only if for FOF_SIMPLEPROGRESS
      };
      if (SHFileOperationA(&fileOp) == 0) {
        return true;
      } else {
        DEBUG_MSG("Delete folder fail! Error code : %d", GetLastError());
      }
    }
  } else {
    // If the directory doesn't exist then just, return true. Behaves like Linux
    if (errno == ENOENT) {
      return true;
    } else {
      DEBUG_MSG("Remove stat fail! Error code : %d", errno);
    }
  }
  return false;
}

//--------------------------------------------------------------------------------------
//   pal::Directory::MakePath
//--------------------------------------------------------------------------------------
bool pal::Directory::makePath(const std::string &path) {
  struct stat st;
  bool rc = false;
  if (path == ".") {
    rc = true;
  } else if (stat(path.c_str(), &st) == 0) {
    if ((st.st_mode & S_IFDIR) != 0) {
      // if a directory path is already exist
      rc = true;
    }
  } else {
    size_t offset = path.find_last_of("/\\");
    if (offset != std::string::npos) {
      std::string newPath = path.substr(0, offset);
      if (!makePath(newPath)) {
        return false;
      }
    }
    pal::Directory::create(path.c_str());
    if ((stat(path.c_str(), &st) == 0) && ((st.st_mode & S_IFDIR) != 0)) {
      rc = true;
    }
  }
  return rc;
}

//---------------------------------------------------------------------------
//    pal::Directory::GetTempDirectory
//---------------------------------------------------------------------------
std::string pal::Directory::getTempDirectory() {
#define TEMP_FOLDER_TEMPLATE "tmpXXXXXX"
  std::string tmpDirPath = std::string();
  char *tmpDirPathArray  = (char *)malloc(MAX_PATH * sizeof(char));

  do {
    if (tmpDirPathArray == NULL) {
      DEBUG_MSG("Fail to get memory!");
      break;
    }

    // get windows temporary path
    if (GetTempPathA(MAX_PATH, tmpDirPathArray) == 0) {
      DEBUG_MSG("Fail to get the path for temporary files! Error code : %d", GetLastError());
      break;
    }

    // Append the template folder name to the temporary path.
    errno_t err =
        strncat_s(tmpDirPathArray, MAX_PATH, TEMP_FOLDER_TEMPLATE, strlen(TEMP_FOLDER_TEMPLATE));
    if (err != 0) {
      DEBUG_MSG("Fail to concatenate the path of tmp directory! Error code: %d", err);
      break;
    }

    // get a unique file name, which will be used as a unique directory name
    err = _mktemp_s(tmpDirPathArray, strlen(tmpDirPathArray) + 1);
    if (err != 0) {
      DEBUG_MSG("Fail to get a unique file name! Error code : %d", err);
      break;
    }
    tmpDirPath = tmpDirPathArray;

    // Create a directory with a unique name in windows temp path
    if (pal::Directory::create(tmpDirPath) == false) {
      DEBUG_MSG("Create temporary directory fail!");
      tmpDirPath.clear();
      break;
    }
  } while (0);

  free(tmpDirPathArray);
  return tmpDirPath;
}

bool pal::Directory::isWritable(const std::string &path) {
  struct stat fileStat;
  if (stat(path.c_str(), &fileStat) != 0) {
    return false; // File not accessible
  }
  return (fileStat.st_mode & _S_IWRITE) != 0; // Check owner write permission
}