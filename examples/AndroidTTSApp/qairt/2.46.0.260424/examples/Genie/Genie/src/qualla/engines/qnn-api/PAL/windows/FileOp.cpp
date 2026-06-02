//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <direct.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <limits.h>
#include <process.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/locking.h>
#include <windows.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "Common.hpp"
#include "PAL/Debug.hpp"
#include "PAL/Directory.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"

// Define a no-op invalid parameter handler
void noopInvalidParameterHandler(const wchar_t *expression,
                                 const wchar_t *function,
                                 const wchar_t *file,
                                 unsigned int line,
                                 uintptr_t pReserved) {
  // Do nothing
}

//-------------------------------------------------------------------------------
//    pal::FileOp::checkFileExists
//-------------------------------------------------------------------------------
bool pal::FileOp::checkFileExists(const std::string &fileName) {
  struct _stat64 st;
  if (_stat64(fileName.c_str(), &st) != 0) {
    DEBUG_MSG("Check File fail! Error code : %d", errno);
    return false;
  }
  return true;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::copyOverFile
//-------------------------------------------------------------------------------
bool pal::FileOp::copyOverFile(const std::string &fromFile, const std::string &toFile) {
  if (CopyFileA(fromFile.c_str(), toFile.c_str(), 0) == 0) {
    DEBUG_MSG("Copy file fail! Error code : %d", errno);
    return false;
  }
  return true;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::move
//-------------------------------------------------------------------------------
bool pal::FileOp::move(const std::string &currentName, const std::string &newName, bool overwrite) {
  struct _stat64 st;
  // if currentName doesn't exist, return false in case newName got deleted
  if (_stat64(currentName.c_str(), &st) != 0) {
    DEBUG_MSG("CurrentName check status fail! Error code : %d", errno);
    return false;
  }
  if (_stat64(newName.c_str(), &st) == 0) {
    if ((st.st_mode & S_IFDIR) != 0) {
      // if newName is directory and overwrite = false, cannot move, return false
      // if newName is directory and overwrite = true, delete it and rename
      if (overwrite == false) {
        return false;
      }
      pal::Directory::remove(newName);
    } else {
      deleteFile(newName);
    }
  }
  // Windows: If newName exist already, rename will return -1. This syscall also fails
  // when the file in question has an open file handle that was not created with
  // shared access/delete permissions
  return (rename(currentName.c_str(), newName.c_str()) == 0);
}

//-------------------------------------------------------------------------------
//    pal::FileOp::deleteFile
//-------------------------------------------------------------------------------
bool pal::FileOp::deleteFile(const std::string &fileName) {
  if (DeleteFileA(fileName.c_str()) == 0) {
    DEBUG_MSG("Delete file fail! Error code : %d", errno);
    return false;
  }
  return true;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::checkIsDir
//-------------------------------------------------------------------------------
bool pal::FileOp::checkIsDir(const std::string &fileName) {
  DWORD result = GetFileAttributesA(fileName.c_str());
  if (result == static_cast<DWORD>(FILE_INVALID_FILE_ID)) {
    DEBUG_MSG("File attribute is invalid_file_id!");
    return false;
  }
  return (result & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getFileInfo
//-------------------------------------------------------------------------------
bool pal::FileOp::getFileInfo(const std::string &filename,
                              pal::FileOp::FilenamePartsType_t &filenameParts) {
  std::string name;
  int32_t lastPathSeparator = std::max(static_cast<int32_t>(filename.find_last_of('\\')),
                                       static_cast<int32_t>(filename.find_last_of('/')));
  if (lastPathSeparator == static_cast<int32_t>(std::string::npos)) {
    // No directory
    name = filename;
  } else {
    // has a directory part
    filenameParts.directory = filename.substr(0, lastPathSeparator);
    name                    = filename.substr(lastPathSeparator + 1);
  }

  size_t ext = name.find_last_of(".");
  if (ext == std::string::npos) {
    // no extension
    filenameParts.basename = name;
  } else {
    // has extension
    filenameParts.basename  = name.substr(0, ext);
    filenameParts.extension = name.substr(ext + 1);
  }
  pal::normalizeSeparator(filenameParts.directory);
  return true;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getFileInfoListRecursiveImpl
//-------------------------------------------------------------------------------
static bool getFileInfoListRecursiveImpl(const std::string &path,
                                         pal::FileOp::FilenamePartsListType_t &filenamePartsList,
                                         const bool ignoreDirs,
                                         size_t maxDepth) {
  // base case
  if (maxDepth == 0) {
    return true;
  }
  if (pal::FileOp::checkIsDir(path) == false) {
    return false;
  }
  int32_t entryCount = 0;
  std::vector<WIN32_FIND_DATAA> nameList;
  entryCount = pal::scanDir(path.c_str(), nameList);
  if (entryCount < 0) {
    return false;
  }
  while (entryCount--) {
    const std::string dName = std::string(nameList[entryCount].cFileName);
    // skip current directory, previous directory and empty string
    if (dName.empty() || dName == "." || dName == "..") {
      continue;
    }
    std::string curPath = path + pal::Path::getSeparator() + dName;
    // recursive if directory but avoid symbolic links to directories
    if (pal::FileOp::checkIsDir(curPath)) {
      struct _stat64 st;
      if (_stat64(curPath.c_str(), &st) == 0 && ((st.st_mode & S_IFDIR) != 0) &&
          (!getFileInfoListRecursiveImpl(curPath, filenamePartsList, ignoreDirs, maxDepth - 1))) {
        return false;
      }
      if (curPath.back() != pal::Path::getSeparator()) {
        curPath += pal::Path::getSeparator();
      }
      // continue here to prevent this object from adding filenameparts in
      // vector but we still need this directory to go recursive
      if (ignoreDirs) {
        continue;
      }
    }
    // add to vector
    pal::FileOp::FilenamePartsType_t filenameParts = {std::string(), std::string(), std::string()};
    if (pal::FileOp::getFileInfo(curPath, filenameParts)) {
      filenamePartsList.push_back(filenameParts);
    }
  }
  return true;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getFileInfoList
//-------------------------------------------------------------------------------
bool pal::FileOp::getFileInfoList(const std::string &path,
                                  FilenamePartsListType_t &filenamePartsList) {
  return getFileInfoListRecursiveImpl(path, filenamePartsList, false, 1);
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getFileInfoListRecursive
//-------------------------------------------------------------------------------
bool pal::FileOp::getFileInfoListRecursive(const std::string &path,
                                           FilenamePartsListType_t &filenamePartsList,
                                           const bool ignoreDirs) {
  return getFileInfoListRecursiveImpl(path, filenamePartsList, ignoreDirs, UINT_MAX);
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getFileVersion
//-------------------------------------------------------------------------------
bool pal::FileOp::getFileVersion(
    const std::string &path, int &major, int &minor, int &teeny, int &build) {
  LPCSTR filePath = path.c_str();
  DWORD handle    = 0;

  // First, confirm the file exists
  if (!pal::FileOp::checkFileExists(path)) {
    DEBUG_MSG("The file does not exist. File: %s", path.c_str());
    return false;
  }

  // Query version info size first
  DWORD infoSize = GetFileVersionInfoSizeA(filePath, &handle);
  if (!infoSize) {
    DEBUG_MSG(
        "Failed to get file version info size. Error code: %lu\n"
        "This might be caused due to absent of version info of the file, "
        "ignore this error and set all version parts to zero.",
        GetLastError());
    major = 0;
    minor = 0;
    teeny = 0;
    build = 0;
    return true;
  }

  // If the size is valid, query the version info data
  std::vector<char> versionInfo(infoSize);
  if (!GetFileVersionInfoA(filePath, handle, infoSize, versionInfo.data())) {
    DEBUG_MSG("Failed to get file version info buffer. Error code: %lu", GetLastError());
    return false;
  }

  LPBYTE buffer = NULL;
  UINT bufSize  = 0;
  // Retrieve the file info from version info data to parse the version numbers
  if (!VerQueryValueA(versionInfo.data(), "\\", reinterpret_cast<LPVOID *>(&buffer), &bufSize)) {
    DEBUG_MSG("Failed to query file info from the buffer. Error code: %lu", GetLastError());
    return false;
  }

  // Parse version data
  VS_FIXEDFILEINFO *fileInfo = reinterpret_cast<VS_FIXEDFILEINFO *>(buffer);
  major                      = HIWORD(fileInfo->dwFileVersionMS);
  minor                      = LOWORD(fileInfo->dwFileVersionMS);
  teeny                      = HIWORD(fileInfo->dwFileVersionLS);
  build                      = LOWORD(fileInfo->dwFileVersionLS);
  return true;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getAbsolutePath
//-------------------------------------------------------------------------------
std::string pal::FileOp::getAbsolutePath(const std::string &path) {
  char fullPath[MAX_PATH];
  if (_fullpath(fullPath, path.c_str(), MAX_PATH) == NULL) {
    DEBUG_MSG("GetAbsolute path fail! Error code : %d", errno);
    return std::string();
  }
  std::string reStr = std::string(fullPath);
  pal::normalizeSeparator(reStr);
  return reStr;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getDirectory
//-------------------------------------------------------------------------------
std::string pal::FileOp::getDirectory(const std::string &file) {
  std::string rc = file;
  int32_t index  = std::max(static_cast<int32_t>(file.find_last_of('\\')),
                           static_cast<int32_t>(file.find_last_of('/')));
  if (index != static_cast<int32_t>(std::string::npos)) {
    rc = file.substr(0, index);
  }
  pal::normalizeSeparator(rc);
  return rc;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::GetFileName
//-------------------------------------------------------------------------------
std::string pal::FileOp::getFileName(const std::string &file) {
  std::string rc = file;
  int32_t index  = std::max(static_cast<int32_t>(file.find_last_of('\\')),
                           static_cast<int32_t>(file.find_last_of('/')));
  if (index != static_cast<int32_t>(std::string::npos)) {
    rc = file.substr(index + 1);  // +1 to skip path separator
  }
  return rc;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::hasFileExtension
//-------------------------------------------------------------------------------
bool pal::FileOp::hasFileExtension(const std::string &file) {
  FilenamePartsType_t parts = {std::string(), std::string(), std::string()};
  getFileInfo(file, parts);
  return !parts.extension.empty();
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getCurrentWorkingDirectory
//-------------------------------------------------------------------------------
std::string pal::FileOp::getCurrentWorkingDirectory() {
  char buffer[MAX_PATH + 1];
  buffer[0] = '\0';

  // If there is any failure return empty string. It is technically possible
  // to handle paths exceeding PATH_MAX on some flavors of *nix but platforms
  // like Android (Bionic) do no provide such capability. For consistency we
  // will not handle extra long path names.
  if (0 == GetCurrentDirectoryA(MAX_PATH, buffer)) {
    DEBUG_MSG("Get current working directory fail! Error code : %d", GetLastError());
    return std::string();
  }
  std::string res = std::string(buffer);
  pal::normalizeSeparator(res);
  return res;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::setCurrentWorkingDirectory
//-------------------------------------------------------------------------------
bool pal::FileOp::setCurrentWorkingDirectory(const std::string &workingDir) {
  return _chdir(workingDir.c_str()) == 0;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::PartsToString
//-------------------------------------------------------------------------------
std::string pal::FileOp::partsToString(const FilenamePartsType_t &filenameParts) {
  std::string path;

  if (!filenameParts.directory.empty()) {
    path += filenameParts.directory;
    path += Path::getSeparator();
  }
  if (!filenameParts.basename.empty()) {
    path += filenameParts.basename;
  }
  if (!filenameParts.extension.empty()) {
    path += ".";
    path += filenameParts.extension;
  }
  pal::normalizeSeparator(path);
  return path;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::open
//-------------------------------------------------------------------------------
int32_t pal::FileOp::open(const std::string &path,
                          pal::FileOp::AccessMode flags,
                          pal::FileOp::FileMode pmode) {
  // Set dwDesiredAccess
  uint32_t dwDesiredAccess = 0;
  if (getFileAccessMode(flags) == pal::FileOp::AccessMode::O_RDONLY_ ||
      getFileAccessMode(flags) == pal::FileOp::AccessMode::O_RDWR_) {
    dwDesiredAccess |= FILE_GENERIC_READ;
  }
  if (getFileAccessMode(flags) == pal::FileOp::AccessMode::O_WRONLY_ ||
      getFileAccessMode(flags) == pal::FileOp::AccessMode::O_RDWR_) {
    dwDesiredAccess |= FILE_GENERIC_WRITE;
  }

  // Set dwShareMode
  uint32_t dwShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;

  // Set dwCreationDisposition
  uint32_t dwCreationDisposition = OPEN_EXISTING;
  if (static_cast<int32_t>(flags & pal::FileOp::AccessMode::O_CREAT_) &&
      !static_cast<int32_t>(flags & pal::FileOp::AccessMode::O_EXCL_)) {
    dwCreationDisposition = OPEN_ALWAYS;
  } else if (static_cast<int32_t>(flags & pal::FileOp::AccessMode::O_TRUNC_)) {
    dwCreationDisposition = CREATE_ALWAYS;
  }

  // Set dwFlagsAndAttributes
  uint32_t dwFlagsAndAttributes = FILE_ATTRIBUTE_NORMAL;
  if (!(static_cast<int32_t>(pmode) &
        ~static_cast<int32_t>(pal::FileOp::FileMode::S_IRUSR_ | pal::FileOp::FileMode::S_IRGRP_ |
                              pal::FileOp::FileMode::S_IROTH_))) {
    // If only read permissions are set, set file permission to readonly
    dwFlagsAndAttributes = FILE_ATTRIBUTE_READONLY;
  }

  // Create the file
  HANDLE fileHandle = CreateFileA(path.c_str(),
                                  dwDesiredAccess,
                                  dwShareMode,
                                  NULL,
                                  dwCreationDisposition,
                                  dwFlagsAndAttributes,
                                  NULL);

  if (fileHandle == INVALID_HANDLE_VALUE) {
    DEBUG_MSG("Could not create file. Error code: %d", errno);
    return -1;
  }

  // Convert the file handle to a file descriptor
  int fd = _open_osfhandle(reinterpret_cast<intptr_t>(fileHandle), _O_WRONLY);
  if (fd == -1) {
    DEBUG_MSG("Could not convert HANDLE to file descriptor! Error code: %d", errno);
    CloseHandle(fileHandle);
    return -1;
  }

  if (static_cast<int32_t>(flags & pal::FileOp::AccessMode::O_APPEND_)) {
    // Move file pointer to the end of the file for append mode
    if (_lseek(fd, 0, SEEK_END) == -1L) {
      DEBUG_MSG("Failed to mode file descriptor pointer to the end for append mode! Error code: %d",
                errno);
      CloseHandle(fileHandle);
      pal::FileOp::close(fd);
      return -1;
    }
  }

  return fd;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::close
//-------------------------------------------------------------------------------
int32_t pal::FileOp::close(int32_t fd) {
  // Prevent program abort when provided invalid file handle
  _set_invalid_parameter_handler(noopInvalidParameterHandler);

  int32_t result = _close(fd);
  if (result != 0) {
    DEBUG_MSG("Close file fail! Error code : %d", errno);
  }

  // Reset invalid file handler to default
  _set_invalid_parameter_handler(nullptr);
  return result;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::totalBytes
//-------------------------------------------------------------------------------
int64_t pal::FileOp::totalBytes(int32_t fd) {
  // Prevent program abort when provided invalid file handle
  _set_invalid_parameter_handler(noopInvalidParameterHandler);

  int64_t bytes = _filelengthi64(fd);

  // Reset invalid file handler to default
  _set_invalid_parameter_handler(nullptr);
  return bytes;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::flock
//-------------------------------------------------------------------------------
int32_t pal::FileOp::flock(int32_t fd, pal::FileOp::FlockOp operation) {
  // Prevent program abort when provided invalid file handle
  _set_invalid_parameter_handler(noopInvalidParameterHandler);

  int32_t opInt = _LK_NBLCK;
  if (operation == pal::FileOp::FlockOp::LOCK_UN_) {
    opInt = _LK_UNLCK;
  }

  // lock/unlock the entire file
  int32_t result = _locking(fd, opInt, LONG_MAX);
  if (result != 0) {
    DEBUG_MSG("Lock file fail! Error code : %d", errno);
  }

  // Reset invalid file handler to default
  _set_invalid_parameter_handler(nullptr);
  return result;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::read
//-------------------------------------------------------------------------------
int32_t pal::FileOp::read(int32_t fd, void *buf, uint32_t count) {
  // Prevent program abort when provided invalid file handle
  _set_invalid_parameter_handler(noopInvalidParameterHandler);

  // Read n-bytes from file to buffer
  int32_t result = _read(fd, buf, count);
  if (result != 0) {
    DEBUG_MSG("Read file error! Error code : %d", errno);
  }

  // Reset invalid file handler to default
  _set_invalid_parameter_handler(nullptr);
  return result;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::write
//-------------------------------------------------------------------------------
int32_t pal::FileOp::write(int32_t fd, const void *buf, uint32_t count) {
  // Prevent program abort when provided invalid file handle
  _set_invalid_parameter_handler(noopInvalidParameterHandler);

  // Write n-bytes from buffer for file
  int32_t result = _write(fd, buf, count);
  if (result != 0) {
    DEBUG_MSG("Write file error! Error code : %d", errno);
  }

  // Reset invalid file handler to default
  _set_invalid_parameter_handler(nullptr);
  return result;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getPid
//-------------------------------------------------------------------------------
int32_t pal::FileOp::getPid() { return static_cast<int32_t>(GetCurrentProcessId()); }

//-------------------------------------------------------------------------------
//    pal::FileOp::makeLink
//-------------------------------------------------------------------------------
bool pal::FileOp::makeLink(const std::string &sourcePath, const std::string &targetPath) {
  return CreateHardLinkA(targetPath.c_str(), sourcePath.c_str(), NULL) != 0;
}

static bool comparePath(wchar_t *pathA, wchar_t *pathB) {
  bool result        = false;
  wchar_t *fullPathA = (wchar_t *)malloc(MAX_PATH * sizeof(wchar_t));
  wchar_t *fullPathB = (wchar_t *)malloc(MAX_PATH * sizeof(wchar_t));
  if (fullPathA != NULL && fullPathB != NULL) {
    if (_wfullpath(fullPathA, pathA, MAX_PATH) != NULL &&
        _wfullpath(fullPathB, pathB, MAX_PATH) != NULL) {
      result = wcscmp(fullPathA, fullPathB) == 0;
    }
  }

  free(fullPathA);
  free(fullPathB);
  return result;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::readLink
//-------------------------------------------------------------------------------
std::string pal::FileOp::readLink(const std::string &linkPath) {
  std::string resultPath = std::string();
  // Read linkPath to get absolute path
  HANDLE hFile = CreateFileA(linkPath.c_str(),
                             GENERIC_READ,
                             FILE_SHARE_READ,
                             NULL,
                             OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL,
                             NULL);
  if (hFile == INVALID_HANDLE_VALUE) {
    CloseHandle(hFile);
    return resultPath;
  }
  const int MAX_PATH_LENGTH  = 512;
  wchar_t *wcharFullLinkPath = (wchar_t *)malloc(MAX_PATH_LENGTH * sizeof(wchar_t));
  if (wcharFullLinkPath == NULL) {
    CloseHandle(hFile);
    free(wcharFullLinkPath);
    return resultPath;
  }
  if (GetFinalPathNameByHandleW(hFile, wcharFullLinkPath, MAX_PATH_LENGTH, VOLUME_NAME_DOS) == 0) {
    free(wcharFullLinkPath);
    return resultPath;
  }
  CloseHandle(hFile);
  // wcharFullLinkPath start with "\\？\c:"
  // We only need to compare the path after drive letter.
  wchar_t *wcharStripLinkPath = wcharFullLinkPath + 6;

  // Get current drive and check whether it is a virtual drive or not.
  // Pass currentDir to QueryDosDeviceA to get deviceName.
  char *currentDir = (char *)malloc(MAX_PATH_LENGTH * sizeof(char));
  if (currentDir == NULL || GetCurrentDirectoryA(MAX_PATH_LENGTH, currentDir) == 0) {
    free(wcharFullLinkPath);
    free(currentDir);
    return resultPath;
  }
  char currentDrive[] = {currentDir[0], ':', '\0'};
  free(currentDir);
  char *deviceName = (char *)malloc(MAX_PATH_LENGTH * sizeof(char));
  if (deviceName == NULL || QueryDosDeviceA(currentDrive, deviceName, MAX_PATH_LENGTH) == 0) {
    free(wcharFullLinkPath);
    free(deviceName);
    return resultPath;
  }
  // Get the mapping length for stripping if it is a virtual drive.
  // If deviceName start with "\??\C:", then it is a virtual drive.
  // Otherwisw, it would start with "\Device\".
  int mappingLen = 0;
  if (deviceName[1] == '?') {
    mappingLen = std::string(deviceName).length() - 6;
  }
  free(deviceName);

  wchar_t *wcharBuf = (wchar_t *)malloc(MAX_PATH_LENGTH * sizeof(wchar_t));
  if (wcharBuf == NULL) {
    free(wcharFullLinkPath);
    free(wcharBuf);
    return resultPath;
  }
  unsigned long longBufSize = 512;
  // This handler can find all the hard links to wcharLinkPath.
  // If any unexpected situation is found in testing, will add more
  // corresponding work around in the future.
  HANDLE currentLink = FindFirstFileNameW(wcharFullLinkPath, 0, &longBufSize, wcharBuf);
  if (currentLink == INVALID_HANDLE_VALUE) {
    free(wcharFullLinkPath);
    free(wcharBuf);
    return resultPath;
  }

  // Successfully get handler
  do {
    // if found file is not the same file of wcharStripLinkPath
    if (wcscmp(wcharBuf, wcharStripLinkPath) != 0) {
      char *foundPath = (char *)malloc(MAX_PATH_LENGTH * sizeof(char));
      if (foundPath != NULL &&
          WideCharToMultiByte(
              CP_ACP, WC_COMPOSITECHECK, wcharBuf, -1, foundPath, MAX_PATH_LENGTH, NULL, NULL) !=
              0) {
        char *mappedFoundPath = foundPath + mappingLen;
        char *fullFoundPath   = (char *)malloc(MAX_PATH_LENGTH * sizeof(char));
        if (fullFoundPath != NULL &&
            _fullpath(fullFoundPath, mappedFoundPath, MAX_PATH_LENGTH) != NULL) {
          resultPath = std::string(fullFoundPath);
          std::replace(resultPath.begin(), resultPath.end(), '\\', '/');
          free(fullFoundPath);
        }
      }

      free(foundPath);
      break;
    }
    // longBufSize will be changed after calling FindFirstFileNameW
    // Might cause error if the file length from FindFirstFileNameW is shorter
    longBufSize = 512;
  } while (FindNextFileNameW(currentLink, &longBufSize, wcharBuf));

  free(wcharFullLinkPath);
  free(wcharBuf);
  FindClose(currentLink);
  return resultPath;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::fileOpen
//-------------------------------------------------------------------------------
FILE *pal::FileOp::fileOpen(const std::string &filename, const std::string &mode) {
  return _fsopen(filename.c_str(), mode.c_str(), _SH_DENYNO);
}

//-------------------------------------------------------------------------------
//    pal::FileOp::strError
//-------------------------------------------------------------------------------
std::string pal::FileOp::strError(int errnum) {
  // cpp stardard or windows doesn't guarantee the max length of strerror or
  // strerror_s. Therefore, we use 256 as max error message length.
  constexpr int maxErrorMsgLength = 256;
  char *buffer                    = (char *)malloc(maxErrorMsgLength * sizeof(char));
  if (buffer == nullptr) {
    DEBUG_MSG("Allocate memory fail for strError.");
    return "";
  }
  std::string errMsg = "";
  if (strerror_s(buffer, maxErrorMsgLength, errnum) == 0) {
    errMsg = buffer;
  } else {
    DEBUG_MSG("Fail to get error message through strError");
  }
  free(buffer);
  return errMsg;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::toLegalFilename
//-------------------------------------------------------------------------------
std::string pal::FileOp::toLegalFilename(const std::string &s) {
  std::string result                           = s;
  const std::unordered_set<char> illegal_chars = {'<', '>', '\"', '\\', '|', '?', ':', '*'};

  for (int i = 0; i < s.size(); ++i) {
    char c = s[i];

    if (illegal_chars.find(c) != illegal_chars.end()) {
      result[i] = '_';
    }
  }

  return result;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getLibraryFileName
//-------------------------------------------------------------------------------
std::string pal::FileOp::getLibraryFileName(const std::string &libName) { return libName + ".dll"; }
