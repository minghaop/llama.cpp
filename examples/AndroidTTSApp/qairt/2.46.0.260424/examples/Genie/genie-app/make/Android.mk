#=============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

LOCAL_PATH := $(call my-dir)
SUPPORTED_TARGET_ABI := arm64-v8a x86 x86_64

#============================ Verify Target Info and Application Variables =========================================
ifneq ($(filter $(TARGET_ARCH_ABI),$(SUPPORTED_TARGET_ABI)),)
    ifneq ($(APP_STL), c++_shared)
        $(error Unsupported APP_STL: "$(APP_STL)")
    endif
else
    $(error Unsupported TARGET_ARCH_ABI: '$(TARGET_ARCH_ABI)')
endif

#============================ Define Common Variables ===============================================================
# Include paths
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../../../../include/Genie
# C++ wrapper headers for genie-app (Genie*.hpp)
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../include
# Include qualla headers (json.hpp)
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../../Genie/src/qualla/include

include $(CLEAR_VARS)
LOCAL_MODULE := libGenie
LOCAL_SRC_FILES := ../../../../lib/aarch64-android/libGenie.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_C_INCLUDES               := $(PACKAGE_C_INCLUDES)

LOCAL_SRC_FILES                := ../main.cpp

LOCAL_MODULE                   := genie-app
LOCAL_SHARED_LIBRARIES         := libGenie
include $(BUILD_EXECUTABLE)
