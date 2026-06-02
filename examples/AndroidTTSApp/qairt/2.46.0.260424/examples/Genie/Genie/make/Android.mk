#=============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
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
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../include
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../../../../include/Genie
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/include
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/include/qualla
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../../../../include/QNN
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../../../../include/QNN/HTP
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/tokenizers
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/MmappedFile/include
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/engines/qnn-api
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/engines/qnn-api/buffer
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/engines/qnn-api/config
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/engines/qnn-api/PAL
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/adaptors
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/dialogs
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/encoders/text-encoders
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/encoders/image-encoders
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/encoder-decoder
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/diffusion-scheduler
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/engines/qnn-cpu
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/engines/qnn-gpu
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/engines/qnn-htp
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/engines/qnn-htp/KVCache
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/qualla/engines/qnn-htp/nsp-utils
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/pipeline
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/trace/include
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/quantization/include
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/resource-manager/include

#========================== Define T2T Lib variables =============================================
include $(CLEAR_VARS)
LOCAL_MODULE := tokenizers_capi
LOCAL_SRC_FILES := ../src/qualla/tokenizers/rust/target/aarch64-linux-android/release/libtokenizers_capi.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_C_INCLUDES               := $(PACKAGE_C_INCLUDES)
MY_SRC_FILES                   := $(wildcard $(LOCAL_PATH)/../src/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/pipeline/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/adaptors/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/dialogs/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/encoders/text-encoders/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/encoders/image-encoders/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/encoder-decoder/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/diffusion-scheduler/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/MmappedFile/src/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engines/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engine-state/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engines/qnn-api/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engines/qnn-api/buffer/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engines/qnn-api/buffer/Allocator/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engines/qnn-api/buffer/Registration/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engines/qnn-api/config/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engines/qnn-api/PAL/linux/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engines/qnn-cpu/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engines/qnn-gpu/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engines/qnn-htp/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engines/qnn-htp/KVCache/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/engines/qnn-htp/nsp-utils/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/utils/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/loggers/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/qualla/samplers/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/trace/src/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/quantization/src/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/resource-manager/src/*.cpp)

LOCAL_MODULE                   := libGenie
LOCAL_SRC_FILES                := $(subst make/,,$(MY_SRC_FILES))
LOCAL_STATIC_LIBRARIES         := tokenizers_capi
LOCAL_LDLIBS                   := -llog
LOCAL_LDFLAGS                  += -Wl,-z,max-page-size=16384
include $(BUILD_SHARED_LIBRARY)
