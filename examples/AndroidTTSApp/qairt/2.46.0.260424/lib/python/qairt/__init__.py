# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import ctypes
import inspect
import os
import platform
import sys
from pkgutil import extend_path as _qairt_extend_path

# DO NOT place any imports at the top-level of this file that are not part of the standard library.
# This is to avoid import errors when running the qairt-vm CLI tool or QAIRT APIS from SDK, as some
# imports may not be available without further checks below.

# Extend the package search path to support namespace packages
__path__ = _qairt_extend_path(__path__, __name__)
__dev_version__, __sdk_version__ = None, None


def _is_oelinux_platform():
    import platform

    if platform.system() != "Linux" or platform.machine() != "aarch64":
        return False

    try:
        with open("/etc/os-release", "r") as f:
            os_release = f.read().lower()
            oe_identifiers = ["qcom", "yocto", "poky", "openembedded", "oe-linux", "ubuntu"]

            for line in os_release.splitlines():
                if any(field in line for field in ["id=", "name=", "pretty_name="]):
                    if any(identifier in line for identifier in oe_identifiers):
                        return True
            return False
    except FileNotFoundError:
        return False


def is_oelinux_user():
    """
    Determines if this import originates from an oelinux/IOT user.
    """
    runtime_only = os.environ.get("ENABLE_QAIRT_RUNTIME_IMPORTS_ONLY", None)
    if runtime_only is not None:
        return runtime_only.lower() == "true"
    return _is_oelinux_platform()


def _load_all_native_libs(lib_dir: str):
    """
    Attempts to load all files with typical shared library extensions in the specified directory.
    This ensures that transitive dependencies are satisfied for subsequent Python extension imports.
    Note: For Windows, it only adds the lib directory to the DLL search path.

    Args:
        lib_dir (str): Path to the directory containing native libraries.

    Raises:
        OSError: If loading a library fails.
    """
    from qairt.cli.qairt_vm.utils.logging import qairt_vm_logger
    from qairt.utils.os_utils import is_windows

    if not lib_dir or not os.path.isdir(lib_dir):
        return
    if is_windows(platform.system()) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(lib_dir)
    for name in sorted(os.listdir(lib_dir)):
        full_path = os.path.join(lib_dir, name)
        lib_ext_to_load = [".so"]
        if not os.path.isfile(full_path) or not any(name.endswith(ext) for ext in lib_ext_to_load):
            continue
        try:
            ctypes.CDLL(full_path)
        except OSError as e:
            qairt_vm_logger.debug(f"Failed loading native lib (load failed): {full_path} -> {e}")


def _setup_qairt_vm_env():
    """
    Set up the QAIRT DEV environment by checking QAIRT SDK installation and dependencies.
    If any issues are found, it attempts to fix them.
    This function also modifies environment variables to ensure the QAIRT SDK is correctly configured.
    If qairt_vm_factory cannot be imported, the function exits silently. (e.g. usage of Python APIs
    directly from QAIRT SDK)
    """
    try:
        import importlib

        from qairt.cli.qairt_vm.helpers.qairt_sdk_helper import get_version_from_path
        from qairt.cli.qairt_vm.qairt_vm_factory import get_platform_qairt_vm
        from qairt.cli.qairt_vm.utils.logging import qairt_vm_logger
    except ImportError:
        return

    # If user already has a QAIRT_SDK_ROOT set, use that
    # Otherwise, use the default SDK root location at install location
    qairt_root_dir = os.environ.get("QAIRT_SDK_ROOT", None)
    qairt_vm = get_platform_qairt_vm(qairt_root_dir=qairt_root_dir)

    # Skip QAIRT VM environment during testing
    is_test_env = any("test" in os.path.basename(frame.filename) for frame in inspect.stack())
    if not is_test_env:
        if len(qairt_vm.check_qairt_dev_env()):
            # Issue warning but don't raise error, as user may choose to ignore and manually fix issues later.
            # This is especially important for users who may have a custom SDK setup that does not meet all
            # checks but is still functional for their use case.
            qairt_vm_logger.warning(
                "QAIRT DEV environment check failed. Please run `qairt-vm -f/--fix` to resolve issues."
            )
    os.environ["QAIRT_SDK_ROOT"] = qairt_vm.qairt_root_dir
    # QNN_SDK_ROOT needs to be set for backwards compatibility till all components remove dependency on it
    os.environ["QNN_SDK_ROOT"] = qairt_vm.qairt_root_dir
    os.environ["PATH"] = qairt_vm.qairt_os_lib_dir + os.pathsep + os.environ["PATH"]
    # Prepend because we want to use qairt python libs before any other
    sys.path.append(qairt_vm.qairt_py_lib_dir)
    # Extend the package search path to support qairt namespace in SDK
    global __path__
    __path__ = _qairt_extend_path(__path__, __name__)
    # Bulk-load all native libs in the SDK OS lib dir
    _load_all_native_libs(qairt_vm.qairt_os_lib_dir)

    global __dev_version__, __sdk_version__
    __dev_version__ = importlib.metadata.version("qairt_dev")
    __sdk_version__ = get_version_from_path(qairt_vm.qairt_root_dir)


# To avoid import errors when running the qairt-vm CLI tool,
# we delay importing qairt until it's actually needed, as some of its imports
# rely on modules from the SDK, which may not be available when running the CLI tool.
qairt_vm_cli_exe = any(
    "qairt-vm" in frame.filename or "qairt_vm" in frame.filename for frame in inspect.stack()
)

should_import_qairt = not qairt_vm_cli_exe

# Updates to public API should be properly guarded to maintain backwards compatibility through qairt-dev
if should_import_qairt:
    _setup_qairt_vm_env()

    from qairt.api._loader import load
    from qairt.api.compiled_model import CompiledModel
    from qairt.api.configs import (
        BackendType,
        Device,
        DeviceInfo,
        DevicePlatformType,
        DspArchitecture,
        ExecutionResult,
        RemoteDeviceIdentifier,
    )
    from qairt.api.executor import ExecutionConfig
    from qairt.api.model import Model
    from qairt.api.profiler import Profiler
    from qairt.modules.cache_module import CacheInfo, CacheModule
    from qairt.modules.dlc_module import DlcModule
    from qairt.utils.asset_utils import AssetType

    if not is_oelinux_user():
        # Preparation APIs are unsupported on oelinux platforms
        from qairt.api.compiler._compile import compile
        from qairt.api.compiler.config import CompileConfig
        from qairt.api.converter._convert import convert
        from qairt.api.converter.converter_config import CalibrationConfig, ConverterConfig

        # Genie execution is unsupported on oelinux platforms
        # ExportFormat was introduced in QAIRT 2.44.0 to support exporting GenAI artifacts
        if __sdk_version__ is not None and __sdk_version__ > "2.43.0":
            from qairt.modules.genie_execution import ExportFormat
