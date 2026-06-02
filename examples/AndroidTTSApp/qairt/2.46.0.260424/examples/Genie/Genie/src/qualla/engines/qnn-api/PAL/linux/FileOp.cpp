//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fcntl.h>

#include <cstdio>
#include <cstdlib>
#if not defined __QNXNTO__ && not defined __hexagon__
#include <sys/sendfile.h>
#endif
#include <dirent.h>

#include <cerrno>
#include <climits>
#ifndef __hexagon__
#include <sys/file.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "PAL/Debug.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"

typedef struct stat Stat_t;

#ifdef __hexagon__
#define PATH_MAX 4096
#endif

//---------------------------------------------------------------------------
//    pal::FileOp::checkFileExists
//---------------------------------------------------------------------------
bool pal::FileOp::checkFileExists(const std::string& fileName) {
  Stat_t sb;

  if (stat(fileName.c_str(), &sb) == -1) {
    return false;
  } else {
    return true;
  }
}

//---------------------------------------------------------------------------
//    pal::FileOp::move
//---------------------------------------------------------------------------
bool pal::FileOp::move(const std::string& currentName,
                       const std::string& newName,
                       const bool overwrite) {
  if (overwrite) {
    remove(newName.c_str());
  }
  return (rename(currentName.c_str(), newName.c_str()) == 0);
}

//---------------------------------------------------------------------------
//    pal::FileOp::deleteFile
//---------------------------------------------------------------------------
bool pal::FileOp::deleteFile(const std::string& fileName) {
  return (remove(fileName.c_str()) == 0);
}

//------------------------------------------------------------------------------
// pal::FileOp::checkIsDir
//------------------------------------------------------------------------------
bool pal::FileOp::checkIsDir(const std::string& fileName) {
  bool retVal = false;
  Stat_t sb;
  if (stat(fileName.c_str(), &sb) == 0) {
    if (sb.st_mode & (unsigned)S_IFDIR) {
      retVal = true;
    }
  }
  return retVal;
}

//------------------------------------------------------------------------------
//    pal::FileOp::getFileInfo
//------------------------------------------------------------------------------
bool pal::FileOp::getFileInfo(const std::string& filename,
                              pal::FileOp::FilenamePartsType_t& filenameParts) {
  std::string name;

  // Clear the result
  filenameParts.basename.clear();
  filenameParts.extension.clear();
  filenameParts.directory.clear();

  const size_t lastPathSeparator = filename.find_last_of(Path::getSeparator());
  if (lastPathSeparator == std::string::npos) {
    // No directory
    name = filename;
  } else {
    // has a directory part
    filenameParts.directory = filename.substr(0, lastPathSeparator);
    name                    = filename.substr(lastPathSeparator + 1);
  }

  const size_t ext = name.find_last_of(".");
  if (ext == std::string::npos) {
    // no extension
    filenameParts.basename = name;
  } else {
    // has extension
    filenameParts.basename  = name.substr(0, ext);
    filenameParts.extension = name.substr(ext + 1);
  }

  return true;
}

#ifndef __hexagon__

//---------------------------------------------------------------------------
//    pal::FileOp::copyOverFile
//---------------------------------------------------------------------------
bool pal::FileOp::copyOverFile(const std::string& fromFile, const std::string& toFile) {
  bool rc = false;
  int readFd;
  int writeFd;
  struct stat statBuf;

  // Open the input file.
  readFd = ::open(fromFile.c_str(), O_RDONLY);
  if (readFd == -1) {
    close(readFd);
    return false;
  }

  // Stat the input file to obtain its size. */
  if (fstat(readFd, &statBuf) != 0) {
    close(readFd);
    return false;
  }

  // Open the output file for writing, with the same permissions as the input
  writeFd = ::open(toFile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, statBuf.st_mode);
  if (writeFd == -1) {
    close(readFd);
    return false;
  }

  // Copy the file in a non-kernel specific way */
  char fileBuf[8192];
  ssize_t rBytes, wBytes;
  while (true) {
    rBytes = read(readFd, fileBuf, sizeof(fileBuf));

    if (!rBytes) {
      rc = true;
      break;
    }

    if (rBytes < 0) {
      rc = false;
      break;
    }

    wBytes = write(writeFd, fileBuf, (size_t)rBytes);

    if (!wBytes) {
      rc = true;
      break;
    }

    if (wBytes < 0) {
      rc = false;
      break;
    }
  }

  /* Close up. */
  close(readFd);
  close(writeFd);
  return rc;
}

static bool getFileInfoListRecursiveImpl(const std::string& path,
                                         pal::FileOp::FilenamePartsListType_t& filenamePartsList,
                                         const bool ignoreDirs,
                                         const size_t maxDepth) {
  struct dirent** namelist = nullptr;
  int entryCount           = 0;

  // Base case
  if (maxDepth == 0) {
    return true;
  }

#ifdef __ANDROID__
  // android dirent.h has the wrong signature for alphasort so it had to be disabled or fixed
  entryCount = scandir(path.c_str(), &namelist, 0, 0);
#else
  entryCount = scandir(path.c_str(), &namelist, 0, alphasort);
#endif
  if (entryCount < 0) {
    return false;
  } else {
    while (entryCount--) {
      const std::string dName(namelist[entryCount]->d_name);
      free(namelist[entryCount]);

      // skip current directory, prev directory and empty string
      if (dName.empty() || dName == "." || dName == "..") {
        continue;
      }

      std::string curPath = path;
      curPath += pal::Path::getSeparator();
      curPath += dName;

      // recurse if directory but avoid symbolic links to directories
      if (pal::FileOp::checkIsDir(curPath)) {
        Stat_t sb;
        if (lstat(curPath.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
          if (!getFileInfoListRecursiveImpl(curPath, filenamePartsList, ignoreDirs, maxDepth - 1)) {
            return false;
          }
        }

        if (ignoreDirs) {
          continue;
        }

        // Append training / to make this path look like a directory for
        // getFileInfo()
        if (curPath.back() != pal::Path::getSeparator()) {
          curPath += pal::Path::getSeparator();
        }
      }

      // add to vector
      pal::FileOp::FilenamePartsType_t filenameParts;
      if (pal::FileOp::getFileInfo(curPath, filenameParts)) {
        filenamePartsList.push_back(filenameParts);
      }
    }

    free(namelist);
  }

  return true;
}

//---------------------------------------------------------------------------
//    pal::FileOp::getFileInfoList
//---------------------------------------------------------------------------
bool pal::FileOp::getFileInfoList(const std::string& path,
                                  FilenamePartsListType_t& filenamePartsList) {
  return getFileInfoListRecursiveImpl(path, filenamePartsList, false, 1);
}

//---------------------------------------------------------------------------
//    pal::FileOp::getFileInfoListRecursive
//---------------------------------------------------------------------------
bool pal::FileOp::getFileInfoListRecursive(const std::string& path,
                                           FilenamePartsListType_t& filenamePartsList,
                                           const bool ignoreDirs) {
  return getFileInfoListRecursiveImpl(
      path, filenamePartsList, ignoreDirs, std::numeric_limits<size_t>::max());
}

//---------------------------------------------------------------------------
//    pal::FileOp::getFileVersion
//---------------------------------------------------------------------------
bool pal::FileOp::getFileVersion(
    const std::string& path, int& major, int& minor, int& teeny, int& build) {
  (void)major;
  (void)minor;
  (void)teeny;
  (void)build;
  // We don't support to get file version in Linux, hence always return true
  DEBUG_MSG("Getting file version in Linux is not supported yet.");
  return true;
}

//---------------------------------------------------------------------------
//    pal::FileOp::getAbsolutePath
//---------------------------------------------------------------------------
std::string pal::FileOp::getAbsolutePath(const std::string& path) {
  // Here we manually implement an algorithm for evaluating absolute path because the realpath()
  // C++11 API fails when the input path DNE and std::filesystem::absolute is gated behind C++17.
  // TODO: update impl to use std::filesystem::absolute when C++ standard is updated to C++17
  if (path.empty()) {
    return pal::FileOp::getCurrentWorkingDirectory();
  }

  std::string unevaluatedPath = path;
  if (path[0] != pal::Path::getSeparator()) {
    unevaluatedPath = pal::Path::combine(getCurrentWorkingDirectory(), path);
  }

  // Simplify path (handle '.' and '..')
  std::vector<std::string> pathComponents = pal::Path::split(unevaluatedPath);
  std::vector<std::string> pathStack;
  for (const std::string& part : pathComponents) {
    if (part.empty() || part == pal::Path::CURRENT_DIR_TOKEN) {
      continue;
    }

    if (part == pal::Path::PARENT_DIR_TOKEN) {
      pathStack.pop_back();
    } else {
      pathStack.push_back(part);
    }
  }

  // Re-assemble parts into the complete absolute path
  return std::accumulate(pathStack.begin(),
                         pathStack.end(),
                         std::string(1, pal::Path::getSeparator()),
                         [](const std::string& a, const std::string& b) -> std::string {
                           return pal::Path::combine(a, b);
                         });
}

//---------------------------------------------------------------------------
//    pal::FileOp::setCWD
//---------------------------------------------------------------------------
bool pal::FileOp::setCurrentWorkingDirectory(const std::string& workingDir) {
  return chdir(workingDir.c_str()) == 0;
}

//---------------------------------------------------------------------------
//    pal::FileOp::Flock
//---------------------------------------------------------------------------
int32_t pal::FileOp::flock(const int32_t fd, const FlockOp operation) {
  return ::flock(fd, static_cast<int32_t>(operation));
}

//---------------------------------------------------------------------------
//    pal::FileOp::makeLink
//---------------------------------------------------------------------------
bool pal::FileOp::makeLink(const std::string& sourcePath, const std::string& targetPath) {
  return symlink(sourcePath.c_str(), targetPath.c_str()) == 0;
}

//---------------------------------------------------------------------------
//    pal::FileOp::readLink
//---------------------------------------------------------------------------
std::string pal::FileOp::readLink(const std::string& linkPath) {
  std::string resultPath = std::string();
  char* tempPath         = (char*)malloc(PATH_MAX * sizeof(char));
  if (tempPath == NULL) {
    return resultPath;
  }
  const int32_t pathSize = readlink(linkPath.c_str(), tempPath, PATH_MAX - 1);

  if (pathSize != -1) {
    tempPath[pathSize] = '\0';
    resultPath         = tempPath;
  }

  free(tempPath);
  return resultPath;
}

#endif  // __hexagon__

//-------------------------------------------------------------------------------
//    pal::FileOp::fileOpen
//-------------------------------------------------------------------------------
FILE* pal::FileOp::fileOpen(const std::string& filename, const std::string& mode) {
  return fopen(filename.c_str(), mode.c_str());
}

//-------------------------------------------------------------------------------
//    pal::FileOp::Write
//-------------------------------------------------------------------------------
#ifndef __HEXAGON_V65__
int32_t pal::FileOp::write(const int32_t fd, const void* buf, const uint32_t count) {
  const ssize_t result = ::write(fd, buf, count);
  if (result == -1) {
    DEBUG_MSG("Write file error! Error code : %d", errno);
  }
  return static_cast<int32_t>(result);
}
#endif

//---------------------------------------------------------------------------
//    pal::FileOp::open
//---------------------------------------------------------------------------
int32_t pal::FileOp::open(const std::string& path, const AccessMode flags, FileMode mode) {
  return ::open(path.c_str(), static_cast<int32_t>(flags), static_cast<uint32_t>(mode));
}

//---------------------------------------------------------------------------
//    pal::FileOp::close
//---------------------------------------------------------------------------
int32_t pal::FileOp::close(const int32_t fd) { return ::close(fd); }

//---------------------------------------------------------------------------
//    pal::FileOp::totalBytes
//---------------------------------------------------------------------------
int64_t pal::FileOp::totalBytes(int32_t fd) {
  // Save the current file position
  const int64_t currentPos = lseek(fd, 0, SEEK_CUR);
  if (currentPos == -1) {
    DEBUG_MSG("Failed to query current file position! Error code : %d", errno);
    return -1;
  }

  // Seek to the end of the file to get the size
  const int64_t fileSize = lseek(fd, 0, SEEK_END);
  if (fileSize == -1) {
    DEBUG_MSG("Encountered an error attempting to calculate file size! Erorr code : %d", errno);
    return -1;
  }

  // Restore the original file position
  if (lseek(fd, currentPos, SEEK_SET) == -1) {
    DEBUG_MSG("Failed to restore file position! Error code : %d", errno);
    return -1;
  }

  return fileSize;
}

//---------------------------------------------------------------------------
//    pal::FileOp::getDirectory
//---------------------------------------------------------------------------
std::string pal::FileOp::getDirectory(const std::string& file) {
  std::string rc      = file;
  const size_t offset = file.find_last_of(Path::getSeparator());
  if (offset != std::string::npos) {
    rc = file.substr(0, offset);
  }
  return rc;
}

//---------------------------------------------------------------------------
//    pal::FileOp::getFileName
//---------------------------------------------------------------------------
std::string pal::FileOp::getFileName(const std::string& file) {
  std::string rc      = file;
  const size_t offset = file.find_last_of(Path::getSeparator());
  if (offset != std::string::npos) {
    rc = file.substr(offset + 1);  // +1 to skip path separator
  }
  return rc;
}

//---------------------------------------------------------------------------
//    pal::FileOp::hasFileExtension
//---------------------------------------------------------------------------
bool pal::FileOp::hasFileExtension(const std::string& file) {
  FilenamePartsType_t parts;
  getFileInfo(file, parts);

  return !parts.extension.empty();
}

//---------------------------------------------------------------------------
//    pal::FileOp::getCWD
//---------------------------------------------------------------------------
std::string pal::FileOp::getCurrentWorkingDirectory() {
  char buffer[PATH_MAX + 1];
  buffer[0] = '\0';

  // If there is any failure return empty string. It is technically possible
  // to handle paths exceeding PATH_MAX on some flavors of *nix but platforms
  // like Android (Bionic) do no provide such capability. For consistency we
  // will not handle extra long path names.
  if (nullptr == getcwd(buffer, PATH_MAX)) {
    return std::string();
  } else {
    return std::string(buffer);
  }
}

//---------------------------------------------------------------------------
//    pal::FileOp::partsToString
//---------------------------------------------------------------------------
std::string pal::FileOp::partsToString(const FilenamePartsType_t& filenameParts) {
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
  return path;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::Read
//-------------------------------------------------------------------------------
int32_t pal::FileOp::read(const int32_t fd, void* buf, const uint32_t count) {
  const ssize_t result = ::read(fd, buf, count);
  if (result == -1) {
    DEBUG_MSG("Read file error! Error code : %d", errno);
  }
  return static_cast<int32_t>(result);
}

//-------------------------------------------------------------------------------
//    pal::FileOp::GetPid
//-------------------------------------------------------------------------------
int32_t pal::FileOp::getPid() { return static_cast<int32_t>(getpid()); }

//-------------------------------------------------------------------------------
//    pal::FileOp::strError
//-------------------------------------------------------------------------------
std::string pal::FileOp::strError(int errnum) { return strerror(errnum); }
//-------------------------------------------------------------------------------
//    pal::FileOp::toLegalFilename
//-------------------------------------------------------------------------------
std::string pal::FileOp::toLegalFilename(const std::string& s) {
  (void)s;
  return s;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getLibraryFileName
//-------------------------------------------------------------------------------
std::string pal::FileOp::getLibraryFileName(const std::string& libName) {
  return "lib" + libName + ".so";
}
