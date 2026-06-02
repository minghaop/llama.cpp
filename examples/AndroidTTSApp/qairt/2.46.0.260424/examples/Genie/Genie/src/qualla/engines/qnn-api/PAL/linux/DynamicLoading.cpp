//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>

#include "PAL/Debug.hpp"
#include "PAL/DynamicLoading.hpp"
#include "PAL/PlatformDetector.hpp"

void *pal::dynamicloading::dlOpen(const char *filename, const int flags) {
  int realFlags = 0;

  if (flags & DL_NOW) {
    realFlags |= RTLD_NOW;
  }

  if (flags & DL_LOCAL) {
    realFlags |= RTLD_LOCAL;
  }

  if (flags & DL_GLOBAL) {
    realFlags |= RTLD_GLOBAL;
  }

  if (flags & DL_NOLOAD) {
#ifndef __hexagon__
    realFlags |= RTLD_NOLOAD;
#else
    DEBUG_MSG("RTLD_NOLOAD is not supported for Hexagon implementation.");
    return NULL;
#endif
  }
  // Try to load the versioned library first
  const char *versionedFilename =
      pal::PlatformDetector::updateLibraryWithVersionSuffix(filename);
  void *handle = ::dlopen(versionedFilename, realFlags);
  /* This conditional logic ensures controlled fallback behavior for loading
   * shared libraries:
   * 1. No retry with unversioned libraries on non-Ubuntu platforms.
   * 2. Retry is attempted only on Ubuntu platforms, and only if dlopen() fails
   * due to "file not found". In all other failure cases (e.g., symbol not
   * found, invalid format), no retry is performed.
   * 3. This fallback mechanism is specifically added to support non-QLI Ubuntu
   * platforms that do not support versioned shared libraries.*/
  if (!handle && versionedFilename != filename) {
    const char *error = ::dlerror();
    // Check if the error indicates file not found
    if (error && (strstr(error, "No such file") ||
                  strstr(error, "cannot open shared object file"))) {
      DEBUG_MSG("Versioned library '%s' not found (%s), falling back to "
                "original '%s'",
                versionedFilename, error, filename);
      handle = ::dlopen(filename, realFlags);
      if (handle) {
        DEBUG_MSG("Successfully loaded library '%s'", filename);
      }
    }
  }

  return handle;
}

void *pal::dynamicloading::dlSym(void *handle, const char *symbol) {
  if (handle == DL_DEFAULT) {
    return ::dlsym(RTLD_DEFAULT, symbol);
  }

  return ::dlsym(handle, symbol);
}

void *pal::dynamicloading::dlAddr(const void *addr) {
  // If the address is empty, return zero as treating failure
  if (!addr) {
    DEBUG_MSG("Input address is nullptr.");
    return nullptr;
  }

  // Dl_info do not maintain the lifetime of its string members,
  // it would be maintained by dlopen() and dlclose(),
  // so we do not need to release it manually
  Dl_info info;
  const int result = ::dladdr(addr, &info);

  if (result) {
    return const_cast<void *>(info.dli_saddr);
  } else {
    DEBUG_MSG("Input address could not be matched to a shared object.");
    return nullptr;
  }
}

int pal::dynamicloading::dlAddrToLibName(const void *addr, std::string &name) {
  // Clean the output buffer
  name = std::string();

  // If the address is empty, return zero as treating failure
  if (!addr) {
    DEBUG_MSG("Input address is nullptr.");
    return 0;
  }

  // Dl_info do not maintain the lifetime of its string members,
  // it would be maintained by dlopen() and dlclose(),
  // so we do not need to release it manually
  Dl_info info;
  const int result = ::dladdr(addr, &info);

  // If dladdr() successes, set name to the library name
  if (result) {
    name = std::string(info.dli_fname);
  } else {
    DEBUG_MSG("Input address could not be matched to a shared object.");
  }

  return result;
}

int pal::dynamicloading::dlClose(void *handle) {
  if (!handle) {
    return 0;
  }

  return ::dlclose(handle);
}

char *pal::dynamicloading::dlError(void) { return ::dlerror(); }
