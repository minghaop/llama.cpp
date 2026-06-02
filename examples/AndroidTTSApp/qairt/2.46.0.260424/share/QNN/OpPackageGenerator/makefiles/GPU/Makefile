# define default
default: all

# define package name
export PACKAGE_NAME := $(notdir $(shell pwd))

# define library prerequisites list
lib_gpu := src
make_dir := makefiles
LIB_SOURCES = $(lib_gpu))

# specify compiler
ifndef CXX
export CXX := clang++
endif

# define default Android ABI
PLATFORM ?= arm64-v8a

.PHONY: all $(LIB_SOURCES) all_android gpu_android
all: $(LIB_SOURCES) all_android

# Android Targets

clean: clean_android clean_bins

clean_bins:
	@rm -rf libs obj

all_android: gpu_android

gpu_android: gpu_aarch64-android

gpu_aarch64-android: check_ndk clean_aarch64-android
	$(call build_if_exists,$(lib_gpu),$(ANDROID_NDK_ROOT)/ndk-build APP_ALLOW_MISSING_DEPS=true APP_ABI="arm64-v8a" NDK_PROJECT_PATH=./ NDK_APPLICATION_MK=$(make_dir)/Application.mk APP_BUILD_SCRIPT=$(make_dir)/Android.mk)
	@$(rename_target_dirs)

clean_android: check_ndk
	-$(ANDROID_NDK_ROOT)/ndk-build clean NDK_PROJECT_PATH=./ NDK_APPLICATION_MK=$(make_dir)/Application.mk APP_BUILD_SCRIPT=$(make_dir)/Android.mk

clean_aarch64-android:
	@rm -rf libs/aarch64-android
	@rm -rf obj/local/aarch64-android

# aarch64-oe-linux-gcc11.2 Target
gpu_oe-11.2: check_esdk
	$(call build_if_exists,$(lib_gpu),-$(MAKE) -f $(make_dir)/Makefile.aarch64-oe-linux-gcc11.2)

clean_oe-11.2:
	@rm -rf libs obj/local/aarch64-oe-linux-gcc11.2 libs/aarch64-oe-linux-gcc11.2

# aarch64-oe-linux-gcc9.3 Target
gpu_oe-9.3: check_esdk
	$(call build_if_exists,$(lib_gpu),-$(MAKE) -f $(make_dir)/Makefile.aarch64-oe-linux-gcc9.3)

clean_oe-9.3:
	@rm -rf libs obj/local/aarch64-oe-linux-gcc9.3 libs/aarch64-oe-linux-gcc9.3

# utilities
# Syntax: $(call build_if_exists <dir>,<cmd>)
build_if_exists = $(if $(wildcard $(1)),$(2),$(warning WARNING: $(1) does not exist. Skipping Compilation))
rename_target_dirs = find . -type d -execdir rename 's/arm64-v8a/aarch64-android/' '{}' \+ \

check_esdk:
ifeq ($(ESDK_ROOT),)
	$(error ERROR: ESDK_ROOT not set, skipping compilation for OE platform(s).)
endif

check_ndk:
ifeq ($(ANDROID_NDK_ROOT),)
	$(error ERROR: ANDROID_NDK_ROOT not set, skipping compilation for Android platform(s).)
endif
