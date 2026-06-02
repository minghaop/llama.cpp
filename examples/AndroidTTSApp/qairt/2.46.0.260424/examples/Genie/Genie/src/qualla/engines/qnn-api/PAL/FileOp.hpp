//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

//------------------------------------------------------------------------------
/// @file
///   This file includes APIs for file operations on the supported platforms
//------------------------------------------------------------------------------

#pragma once

#include <fcntl.h>

#include <cstdint>
#include <string>
#include <vector>

namespace pal {

//------------------------------------------------------------------------------
/// @brief
///   FileOp contains OS Specific file system functionality.
//------------------------------------------------------------------------------
struct FileOp {
  // enum for file operation mode, strictly follow linux usage
  // windows or another OS user should transfer the usage
  enum class AccessMode : int32_t {
    O_RDONLY_ = O_RDONLY,  // File access flag: Read only
    O_WRONLY_ = O_WRONLY,  // File access flag: Write only
    O_RDWR_   = O_RDWR,    // File access flag: Read and write
    O_CREAT_  = O_CREAT,   // File creation flag: Create file or open existing
    O_EXCL_   = O_EXCL,    // File creation flag: Opens file, creates if DNE (use with O_CREAT)
    O_TRUNC_  = O_TRUNC,   // File creation flag: Truncate file on open
    O_APPEND_ = O_APPEND   // File status flag: Open file and shift fp to end of file
  };

  friend AccessMode operator&(AccessMode lhs, AccessMode rhs) {
    return static_cast<AccessMode>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
  }
  friend AccessMode operator|(AccessMode lhs, AccessMode rhs) {
    return static_cast<AccessMode>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
  }
  static AccessMode getFileAccessMode(AccessMode mode) {
    return mode & (AccessMode::O_RDONLY_ | AccessMode::O_WRONLY_ | AccessMode::O_RDWR_);
  }

  // enum for symbolic constants mode, strictly follow linux usage
  // windows or another OS user should transfer the usage
  // ref : http://man7.org/linux/man-pages/man2/open.2.html
  enum class FileMode : uint32_t {
    S_DEFAULT_ = 0777,  // Full permission
    S_IRWXU_   = 0700,  // User read, write, exec
    S_IRUSR_   = 0400,  // User read
    S_IWUSR_   = 0200,  // User write
    S_IXUSR_   = 0100,  // User exec
    S_IRWXG_   = 0070,  // Group read, write, exec
    S_IRGRP_   = 0040,  // Group read
    S_IWGRP_   = 0020,  // Group write
    S_IXGRP_   = 0010,  // Group exec
    S_IRWXO_   = 0007,  // Other read, write, exec
    S_IROTH_   = 0004,  // Other read
    S_IWOTH_   = 0002,  // Other write
    S_IXOTH_   = 0001   // Other exec
  };

  friend FileMode operator&(FileMode lhs, FileMode rhs) {
    return static_cast<FileMode>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
  }
  friend FileMode operator|(FileMode lhs, FileMode rhs) {
    return static_cast<FileMode>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
  }

  // enum for flock operation, strictly follow linux usage
  // windows or another OS user should transfer the usage
  enum class FlockOp : int32_t { LOCK_SH_ = 1, LOCK_EX_ = 2, LOCK_NB_ = 4, LOCK_UN_ = 8 };

  friend FlockOp operator|(FlockOp lhs, FlockOp rhs) {
    return static_cast<FlockOp>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
  }

  //---------------------------------------------------------------------------
  /// @brief
  ///   Open a file
  /// @param path, flags
  ///   Path to check, flags to set permissions
  /// @return
  ///   Returns a file descriptor or -1 to indicate a failure
  ///
  /// For examples:
  /// -# open a write/read file:
  ///    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ///    int32_t fd = pal::FileOp::open(path, pal::FileOp::AccessMode::O_RDWR_);
  ///    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  /// -# open a read-only file:
  ///    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ///    int32_t fd = pal::FileOp::open(path, pal::FileOp::AccessMode::O_RDONLY_);
  ///    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  /// -# open a write only file with append:
  ///    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ///    int32_t fd = pal::FileOp::open(path, pal::FileOp::AccessMode::O_WRONLY_ |
  ///                                         pal::FileOp::AccessMode::O_APPEND);
  ///    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  /// -# open/create a new file with user write/read/exec + other read only
  ///    + group read only
  ///    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ///    int32_t fd = pal::FileOp::open(path, pal::FileOp::AccessMode::O_CREAT_ |
  ///    pal::FileOp::AccessMode::O_RDWR_, pal::FileOp::FileMode::S_IRWXU_ |
  ///    pal::FileOp::FileMode::S_IRGRP_ | pal::FileOp::FileMode::S_IROTH_);
  ///    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ///    In this case, linux can work as expected. Windows will creat a file with
  ///    both read and write since at least one of three kinds can read.
  //--------------------------------------------------------------------------
  static int32_t open(const std::string &path,
                      AccessMode flags,
                      FileMode mode = FileMode::S_DEFAULT_);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Closes a file
  /// @param fd
  ///   File descriptor
  /// @return
  ///   Returns 0 if successful, -1 otherwise
  //---------------------------------------------------------------------------
  static int32_t close(int32_t fd);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Calculate the total size of the given file. This function does not alter
  ///   the client's current position in the file descriptor
  /// @param fd
  ///   File descriptor
  /// @return
  ///   Returns Total size of the file in bytes, -1 upon error
  //---------------------------------------------------------------------------
  static int64_t totalBytes(int32_t fd);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Write a file
  /// @param fd
  ///   File descriptor
  /// @param buf
  ///   Data to be written
  /// @param count
  ///   Number of bytes
  /// @return
  ///   Number of bytes or -1 to indicate failure
  ///
  /// @note Please include <errno.h> if you need to use errno
  /// @todo Both Linux and Windows can use almost the same errno MACRO
  ///       will discuss further error handling or unified errno interface
  //---------------------------------------------------------------------------
  static int32_t write(int32_t fd, const void *buf, uint32_t count);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Opens a file.
  /// @param filename
  ///   File name.
  /// @param mode
  ///   Kind of access that's enabled.
  /// @return
  ///   Return a FILE pointer. Otherwise, nullptr is returned and errno is set
  ///   to indicate the error.
  //---------------------------------------------------------------------------
  static FILE *fileOpen(const std::string &filename, const std::string &mode);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Copies a file from one location to another, overwrites if the
  ///   destination already exists.
  /// @param source
  ///   File name of the source file.
  /// @param target
  ///   File name of the target file.
  /// @return
  ///   True on success, otherwise false.
  //---------------------------------------------------------------------------
  static bool copyOverFile(const std::string &source, const std::string &target);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Checks whether the file or directory exists or not.
  /// @param fileName
  ///   File's or directory's full path
  /// @return
  ///   True on success, otherwise false.
  //---------------------------------------------------------------------------
  static bool checkFileExists(const std::string &fileName);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Renames an existing file. If the file with target name exists, this call
  ///   overwrites it with the file with source name.
  /// @param source
  ///   Current File name.
  /// @param target
  ///   New name of the file.
  /// @param overwrite
  ///   Flag indicating to overwrite existing file with newName
  /// @return
  ///   True if successful, otherwise false.
  ///
  /// @note Does not work if source and target are on different filesystems. In
  ///       such cases, opt for copyOverFile() instead
  /// @note Windows: If there exist open file handles for the target file, then
  ///       deleteFile is expected to fail as long as that existing file handle
  ///       was not created with shared access/delete permissions.
  //---------------------------------------------------------------------------
  static bool move(const std::string &source, const std::string &target, bool overwrite);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Delete an existing file.
  /// @param fileName
  ///   File name of the file to be deleted.
  /// @return
  ///   True if successful, otherwise false.
  ///
  /// @note Windows: If there exist open file handles for the target file, then
  ///       deleteFile is expected to fail as long as that existing file handle
  ///       was not created with shared access/delete permissions.
  //---------------------------------------------------------------------------
  static bool deleteFile(const std::string &fileName);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Check if path is a directory or not
  /// @param path
  ///   Path to check
  /// @return
  ///   True if successful, otherwise false.
  //---------------------------------------------------------------------------
  static bool checkIsDir(const std::string &path);

  //---------------------------------------------------------------------------
  /// @brief Data type representing parts of a filename
  //---------------------------------------------------------------------------
  typedef struct {
    //---------------------------------------------------------------------------
    /// @brief Name of the file without the extension (i.e., basename)
    //---------------------------------------------------------------------------
    std::string basename;

    //---------------------------------------------------------------------------
    /// @brief Name of the file extension (i.e., .txt or .hlnd, .html)
    //---------------------------------------------------------------------------
    std::string extension;

    //---------------------------------------------------------------------------
    /// @brief
    ///   Location of the file (i.e., /abc/xyz/foo.bar <-- /abc/xyz/).
    ///   If the file name has no location then the Directory points to
    ///   empty string
    //---------------------------------------------------------------------------
    std::string directory;
  } FilenamePartsType_t;

  //---------------------------------------------------------------------------
  /// @brief
  ///   Determines the components of a given filename, being the directory,
  ///   basename and extension. If the file has no location or extension, these
  ///   components remain empty
  /// @param filename
  ///   Path of the file for which the components are to be determined
  /// @param filenameParts
  ///   Will contain the file name components when this function returns
  /// @return
  ///   True if successful, false otherwise
  //---------------------------------------------------------------------------
  static bool getFileInfo(const std::string &filename, FilenamePartsType_t &filenameParts);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Typedef for a vector of FilenamePartsType_t
  //---------------------------------------------------------------------------
  typedef std::vector<FilenamePartsType_t> FilenamePartsListType_t;

  //---------------------------------------------------------------------------
  /// @brief
  ///   Typedef for a vector of FilenamePartsType_t const iterator
  //---------------------------------------------------------------------------
  typedef std::vector<FilenamePartsType_t>::const_iterator FilenamePartsListTypeIter_t;

  //---------------------------------------------------------------------------
  /// @brief
  ///   Returns a vector of FilenamePartsType_t objects for a given directory
  /// @param path
  ///   Path to scan for files
  /// @return
  ///   True if successful, false otherwise
  //---------------------------------------------------------------------------
  static bool getFileInfoList(const std::string &path, FilenamePartsListType_t &filenamePartsList);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Returns a vector of FilenamePartsType_t objects for a given directory
  ///   and the child directories inside.
  /// @param path
  ///   Path to directory to scan for files for
  ///   @note if path is not a directory - the function will return false
  /// @param filenamePartList
  ///   List to append to
  /// @param ignoreDirs
  ///   If this flag is set to true, directories (and symbolic links to directories)
  ///   are not included in the list. Only actual files below the specified
  ///   directory path will be appended.
  /// @return True if successful, false otherwise
  /// @note Directories in list only populate Directory member variable of the struct.
  ///       That is Basename and Extension will be empty strings.
  /// @note Symbolic links to directories are not followed. This is to avoid possible
  ///       infinite recursion. However the initial call to this method can have
  ///       path to be a symbolic link to a directory. If ignoreDirs is true,
  ///       symbolic links to directories are also ignored.
  /// @note The order in which the files/directories are listed is platform
  ///       dependent. However files inside a directory always come before the
  ///       directory itself.
  //---------------------------------------------------------------------------
  static bool getFileInfoListRecursive(const std::string &path,
                                       FilenamePartsListType_t &filenamePartsList,
                                       const bool ignoreDirs);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Get the file version numbers. The pattern is <major>.<minor>.<teeny>.<build>
  /// @param path
  ///   Path to the file to be parsed
  /// @param major
  ///   Major version number(the first number)
  /// @param minor
  ///   Minor version number(the second number)
  /// @param teeny
  ///   Teeny number(the third number)
  /// @param build
  ///   Build number(the fourth number)
  /// @return
  ///   True if 2 cases happen:
  ///   1. Parse file version successfully.
  ///   2. File exists but failed to parse version info, return zero-values version and return true.
  ///   False if 2 cases happen:
  ///   1. File does not exist.
  ///   2. If any error occurs when parsing a queried version info table.
  //---------------------------------------------------------------------------
  static bool getFileVersion(
      const std::string &path, int &major, int &minor, int &teeny, int &build);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Create an absolute path from the supplied path
  /// @param path
  ///   Path should not contain trailing '/' or '\\'
  /// @return
  ///   Return absolute path without trailing '/' or '\\'
  //---------------------------------------------------------------------------
  static std::string getAbsolutePath(const std::string &path);

  //---------------------------------------------------------------------------
  /// @brief Get the file name from a path
  //---------------------------------------------------------------------------
  static std::string getFileName(const std::string &file);

  //---------------------------------------------------------------------------
  /// @brief Get the directory path to a file
  //---------------------------------------------------------------------------
  static std::string getDirectory(const std::string &file);

  //---------------------------------------------------------------------------
  /// @brief Get the current working directory.
  /// @returns The absolute CWD or empty string if the path could not be
  ///          retrieved (because it was too long or deleted for example).
  //---------------------------------------------------------------------------
  static std::string getCurrentWorkingDirectory();

  //---------------------------------------------------------------------------
  /// @brief Set the current working directory
  //---------------------------------------------------------------------------
  static bool setCurrentWorkingDirectory(const std::string &workingDir);

  //---------------------------------------------------------------------------
  /// @brief Returns true if the file contains any extension or false.
  //---------------------------------------------------------------------------
  static bool hasFileExtension(const std::string &file);

  //---------------------------------------------------------------------------
  /// @brief Returns full path of file, Directory/Basename(.Extension, if any)
  //---------------------------------------------------------------------------
  static std::string partsToString(const FilenamePartsType_t &filenameParts);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Locks a file
  /// @param fd
  ///   File descriptor
  /// @return
  ///   Returns 0 if successful, -1 otherwise
  ///
  /// For example:
  /// -# place an exclusive lock
  ///    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ///    pal::FileOp::Flock(fd, pal::FileOp::FlockOp::LOCK_EX_);
  ///    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  /// -# unlock a file
  ///    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ///    pal::FileOp::Flock(fd, pal::FileOp::FlockOp::LOCK_UN_);
  ///    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ///
  /// @note This API doesn't suppport shared_lock since windows doesn't have the option
  ///       and current linux doesn't use shared lock as well.
  /// @todo Allow shared lock in necessary in the future.
  //---------------------------------------------------------------------------
  static int32_t flock(int32_t fd, FlockOp operation);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Read a file
  /// @param fd
  ///   File descriptor
  /// @param buf
  ///   Storage location for data
  /// @param count
  ///   Number of bytes
  /// @return
  ///   1. Number of bytes
  ///   2. 0, indicates end of file
  ///   3. -1, indicates failure
  ///
  /// @note Please include <errno.h> if you need to use errno
  /// @todo Both Linux and Windows can use almost the same errno MACRO
  ///       will discuss further error handling or unified errno interface
  //---------------------------------------------------------------------------
  static int32_t read(int32_t fd, void *buf, uint32_t count);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Get process id
  /// @return
  ///   Return process id
  //---------------------------------------------------------------------------
  static int32_t getPid(void);

  //---------------------------------------------------------------------------
  /// @brief
  ///   On Linux, create a "symbolic link" from a source path.
  ///   On Windows, create a "hard link" from a source path. It is a work aound
  ///   because admin privilege is required to create a symbolic link on
  ///   Windows, and using hard link should not cause significant different
  ///   on current implementation.
  ///
  ///   Note that on Windows, current implementation cannot create a hard link for
  ///   a directory or without original file.
  ///   If more similar symlink functionality is required should study reparse point
  /// @param sourcePath
  ///   The location of the source file.
  /// @param targetPath
  ///   The location of the output file of created link.
  /// @return
  ///   True on success, otherwise false.
  //---------------------------------------------------------------------------
  static bool makeLink(const std::string &sourcePath, const std::string &targetPath);

  //---------------------------------------------------------------------------
  /// @brief
  ///   On Linux, get the readlink result from a "symbolic link".
  ///   On Windows, pair with the work around of pal::FileOp::MakeLink,
  ///   get "full"(absolute) original file path from a "hard link".
  /// @param linkPath
  ///   File path of a symbolic (on Linux)/ hard (on Windows) link.
  /// @return
  ///   The path of contents/original file of the symbolic/hard link when found,
  ///   otherwise an empty string
  //---------------------------------------------------------------------------
  static std::string readLink(const std::string &linkPath);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Describes the error code passed in the argument errnum.
  /// @param errnum
  ///   Error number.
  /// @return
  ///   The appropriate error description string
  //---------------------------------------------------------------------------
  static std::string strError(int errnum);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Remove the illegal char in filename and replace with "_".
  ///   On Linux, return the string directly.
  ///   On Windows, replace the illegal string type:
  ///   '<', '>', '\"', '\\','|', '?', ':', '*' with '_'.
  /// @param s
  ///   File name to be checked.
  /// @return
  ///   The appropriate string of filename.
  //---------------------------------------------------------------------------
  static std::string toLegalFilename(const std::string &s);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Get the full name of the library file with platform-specific prefix
  ///   and extension.
  ///   On Linux, prepends "lib" and appends ".so" (e.g., "QnnHtp" -> "libQnnHtp.so")
  ///   On Windows, appends ".dll" (e.g., "QnnHtp" -> "QnnHtp.dll")
  /// @param libName
  ///   The base name of the library file without platform-specific prefix or extension.
  /// @return
  ///   The full name of the library file with appropriate platform-specific
  ///   prefix and extension.
  //---------------------------------------------------------------------------
  static std::string getLibraryFileName(const std::string &libName);
};

}  // namespace pal
