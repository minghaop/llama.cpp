# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


def is_arm_64(arch_name):
    return any(name in arch_name.lower() for name in ["aarch64", "arm64", "armv8"])


def is_x86_64(arch_name):
    return any(name in arch_name.lower() for name in ["x86_64", "amd64", "intel64"])


def is_windows(platform_name):
    return platform_name.lower() == "windows"


def is_linux(platform_name):
    return platform_name.lower() == "linux"
