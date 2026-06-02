# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import os
import tempfile
from typing import Dict, Optional, Union
from uuid import uuid4

from qairt.api.configs.common import BackendType
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    ConnectionType,
    DeviceEnvironmentContext,
    DeviceInfo,
    DevicePlatformType,
)
from qti.aisw.tools.core.utilities.devices.api.device_factory import DeviceFactory
from qti.aisw.tools.core.utilities.devices.api.device_interface import DeviceInterface
from qti.aisw.tools.core.utilities.devices.utils.device_code import (
    DeviceCode,
    DeviceCompletedProcess,
    DeviceFailedProcess,
    DeviceReturn,
)


class GenieCLIMixin:
    """
    Mixin class providing common functionality for Genie CLI-based runners.

    Handles:
    - Device setup and management
    - Artifact deployment and cleanup
    - SDK artifact management
    - Device environment configuration
    - Common error handling patterns
    """

    # Target root directories for different platforms
    _TARGET_ROOTS = {
        DevicePlatformType.ANDROID: "/data/local/tmp/",
        DevicePlatformType.LINUX_EMBEDDED: "/data/local/tmp",
        DevicePlatformType.X86_64_LINUX: os.getenv("QAIRT_TMP_DIR", default=tempfile.gettempdir()),
    }

    _FASTRPC_SKEL_DIR = "/usr/share/fastrpc"

    _SUPPORTED_PLATFORMS = list(_TARGET_ROOTS.keys())

    def _init_genie_cli_mixin(
        self,
        backend: BackendType,
        device: Union[DeviceInfo, DeviceInterface],
        qairt_sdk_root: Optional[Union[str, os.PathLike]] = None,
        clean_up: bool = False,
    ):
        """Initialize the mixin with common Genie CLI functionality.

        This should be called from the __init__ method of classes using this mixin.

        Args:
            backend: The backend type for execution
            device: Device to execute on
            qairt_sdk_root: Path to QAIRT SDK
            clean_up: Whether to clean up artifacts on destruction
        """
        self._backend = backend
        self._device = device if isinstance(device, DeviceInterface) else DeviceFactory.create_device(device)
        assert self._device.device_info is not None, f"Could not set device info for {self._device}"

        self._target_sep = "/" if self._device.device_info.platform_type != DevicePlatformType.WOS else "\\"
        self._target_root = (
            self._TARGET_ROOTS[self._device.device_info.platform_type] + self._target_sep + str(uuid4())
        )

        self._target_root_exists = False
        self._clean_up = clean_up
        self._config_artifacts: Dict[str | os.PathLike, str | os.PathLike] = {}
        self._config_artifacts_loaded = False
        self._sdk_artifacts_loaded = False

        # Handle QAIRT SDK root
        if qairt_sdk_root:
            if os.path.isdir(qairt_sdk_root):
                self._qairt_sdk_root = qairt_sdk_root
            else:
                raise NotADirectoryError(f"Provided QAIRT SDK root is not a directory: {qairt_sdk_root}")
        else:
            self._qairt_sdk_root = str(os.environ.get("QNN_SDK_ROOT"))

    @classmethod
    def _check_device_return(
        cls, device_return: DeviceReturn, error_prefix: str = ""
    ) -> DeviceCompletedProcess:
        """Check device return status and raise appropriate errors.

        Args:
            device_return: The return value from a device operation
            error_prefix: Optional prefix for error messages

        Returns:
            DeviceCompletedProcess: The completed process if successful

        Raises:
            RuntimeError: If the device operation failed
        """
        if isinstance(device_return, DeviceFailedProcess):
            raise RuntimeError(error_prefix + f"Original Error: {device_return.orig_error}")
        if (
            isinstance(device_return, DeviceCompletedProcess)
            and device_return.returncode != DeviceCode.DEVICE_SUCCESS
        ):
            raise RuntimeError(
                error_prefix + f"return code: {device_return.returncode}\tstderr: {device_return}"
            )
        return device_return

    def _ensure_target_root_exists(self):
        """Ensure the target root directory exists on the device."""
        if not self._target_root_exists:
            self._check_device_return(
                self._device.make_directory(self._target_root),
                f"Failed to make directory {self._target_root} on target\t",
            )
            self._target_root_exists = True

    def _add_config_artifact(self, artifact_path: str | os.PathLike) -> str | os.PathLike:
        """Add config artifact for deployment to device.

        Args:
            artifact_path: Path to the artifact file to be deployed

        Returns:
            str | os.PathLike: Target path where the artifact will be deployed on the device
        """
        assert self._device.device_info is not None

        # For local execution on non-Android platforms, use the original path
        if (
            self._device.device_info.connection_type == ConnectionType.LOCAL
            and self._device.device_info.platform_type not in self._SUPPORTED_PLATFORMS
        ):
            return artifact_path

        # Generate unique target path to avoid conflicts
        target_path = self._target_sep.join([self._target_root, os.path.basename(artifact_path)])
        while target_path in self._config_artifacts:
            target_path = self._target_sep.join(
                [self._target_root, str(uuid4())[:6] + os.path.basename(artifact_path)]
            )

        self._config_artifacts[target_path] = artifact_path
        return target_path

    def _push_config_artifacts(self):
        """Push all registered config artifacts to the device."""
        if not self._config_artifacts_loaded:
            for dst, src in self._config_artifacts.items():
                self._check_device_return(
                    self._device.push(src, dst), f"Failed to push artifact {src} to device\t"
                )
            self._config_artifacts_loaded = True

    def _target_qairt_dir(self) -> str:
        """Get target QAIRT directory path.

        Returns:
            str: Path to the QAIRT directory on the target device
        """
        assert self._device.device_info is not None

        target_name = ""
        if self._device.device_info.platform_type == DevicePlatformType.ANDROID:
            target_name = "aarch64-android"
        elif self._device.device_info.platform_type == DevicePlatformType.X86_64_LINUX:
            target_name = "x86_64-linux-clang"
        elif self._device.device_info.platform_type == DevicePlatformType.LINUX_EMBEDDED:
            target_name = "aarch64-oe-linux-gcc11.2"

        # For local execution on non-Android platforms, use SDK bin directory
        if (
            self._device.device_info.connection_type == ConnectionType.LOCAL
            and self._device.device_info.platform_type not in self._SUPPORTED_PLATFORMS
        ):
            return self._target_sep.join([str(self._qairt_sdk_root), "bin", target_name])

        return self._target_sep.join([self._target_root])

    def _get_target_dir(self, dir_type: str) -> str:
        """Get target directory path for lib or bin (T2T style with subdirectories).

        Args:
            dir_type: Either "lib" or "bin"

        Returns:
            Path to the target directory
        """
        assert self._device.device_info is not None

        if self._device.device_info.connection_type == ConnectionType.LOCAL:
            target_name = ""
            if self._device.device_info.platform_type == DevicePlatformType.X86_64_LINUX:
                target_name = "x86_64-linux-clang"
            return self._target_sep.join([str(self._qairt_sdk_root), dir_type, target_name])

        return self._target_sep.join([self._target_root, dir_type])

    def _target_lib_dir(self) -> str:
        """Get target library directory path (for T2T style)."""
        return self._get_target_dir("lib")

    def _target_bin_dir(self) -> str:
        """Get target binary directory path (for T2T style)."""
        return self._get_target_dir("bin")

    def _push_skel_to_fastrpc(self, skel_libs: list[str]) -> None:
        """Push skel libs to /usr/share/fastrpc on LINUX_EMBEDDED devices.

        Creates the directory if it does not exist, then pushes each skel lib.
        Skips libs that do not exist on the host.

        Args:
            skel_libs: List of local paths to skel .so files.
        """
        mkdir_return = self._device.make_directory(self._FASTRPC_SKEL_DIR)
        if isinstance(mkdir_return, DeviceFailedProcess):
            if hasattr(self, "_logger") and self._logger:
                self._logger.warning(
                    f"Could not create {self._FASTRPC_SKEL_DIR} on device: "
                    f"{mkdir_return.orig_error}. Skipping fastrpc skel push."
                )
            return

        if hasattr(self, "_logger") and self._logger:
            self._logger.warning(
                f"Pushing skel libs to {self._FASTRPC_SKEL_DIR} on device. "
                "These files will NOT be removed during cleanup — manual removal may be required."
            )

        for skel_lib in skel_libs:
            if not os.path.exists(skel_lib):
                continue
            self._check_device_return(
                self._device.push(skel_lib, self._FASTRPC_SKEL_DIR),
                f"Failed to push skel lib to fastrpc: {skel_lib}\t",
            )

    def _push_sdk_artifacts(self, binary_name: str = "genie-t2t-run", use_subdirectories: bool = False):
        """Push SDK artifacts to device.

        Args:
            binary_name: Name of the primary binary to deploy (e.g., "genie-app", "genie-t2t-run")
            use_subdirectories: If True, use lib/ and bin/ subdirectories (T2T style).
                               If False, put everything in root (App style)
        """
        from qairt.modules.genie_execution.genie_t2t_runner_helper import GenieT2TRunnerHelper

        # Get SDK artifacts needed for the target device and backend
        bins_to_push, libs_to_push = GenieT2TRunnerHelper.get_sdk_artifacts(
            self._device, self._qairt_sdk_root, self._backend
        )

        # Create directories for SDK artifacts
        if use_subdirectories:
            lib_dir = self._target_sep.join([self._target_root, "lib"])
            bin_dir = self._target_sep.join([self._target_root, "bin"])
            self._check_device_return(
                self._device.make_directory(lib_dir),
                f"Failed to make lib directory {lib_dir} on target\t",
            )
            self._check_device_return(
                self._device.make_directory(bin_dir),
                f"Failed to make bin directory {bin_dir} on target\t",
            )
        else:
            # Single directory (GenieApp style)
            qairt_dir = self._target_sep.join([self._target_root])
            self._check_device_return(
                self._device.make_directory(qairt_dir),
                f"Failed to make qairt directory {qairt_dir} on target\t",
            )

        # Push library files
        for lib_artifact in libs_to_push:
            if os.path.exists(lib_artifact):
                if use_subdirectories:
                    target_path = self._target_sep.join(
                        [self._target_root, "lib", os.path.basename(lib_artifact)]
                    )
                else:
                    target_path = self._target_sep.join([self._target_root, os.path.basename(lib_artifact)])
                self._check_device_return(
                    self._device.push(lib_artifact, target_path),
                    f"Failed to push library: {lib_artifact}\t",
                )

        # Push binary files with optional name replacement
        for bin_artifact in bins_to_push:
            if os.path.exists(bin_artifact):
                # Replace default binary name if different binary is requested
                if binary_name != "genie-t2t-run" and "genie-t2t-run" in bin_artifact:
                    bin_artifact = bin_artifact.replace("genie-t2t-run", binary_name)

                if os.path.exists(bin_artifact):
                    if use_subdirectories:
                        target_path = self._target_sep.join(
                            [self._target_root, "bin", os.path.basename(bin_artifact)]
                        )
                    else:
                        target_path = self._target_sep.join(
                            [self._target_root, os.path.basename(bin_artifact)]
                        )
                    self._check_device_return(
                        self._device.push(bin_artifact, target_path),
                        f"Failed to push binary: {bin_artifact}\t",
                    )
                else:
                    # Use logger from derived class if available
                    if hasattr(self, "_logger") and self._logger:
                        self._logger.warning(f"Binary not found: {bin_artifact}")

        # For LINUX_EMBEDDED, also push skel libs to /usr/share/fastrpc
        assert self._device.device_info is not None
        if self._device.device_info.platform_type == DevicePlatformType.LINUX_EMBEDDED:
            skel_libs = [lib for lib in libs_to_push if "Skel.so" in os.path.basename(lib)]
            if skel_libs:
                self._push_skel_to_fastrpc(skel_libs)

        self._sdk_artifacts_loaded = True

    def _get_device_environment(self, use_subdirectories: bool = False) -> DeviceEnvironmentContext:
        """Get device environment context.

        Args:
            use_subdirectories: If True, use lib/ directory for library path (T2T style).
                               If False, use root directory (App style)

        Returns:
            DeviceEnvironmentContext: Environment configuration for device execution
        """
        env = DeviceEnvironmentContext()
        env.cwd = self._target_root

        assert self._device.device_info is not None
        if self._device.device_info.platform_type in self._SUPPORTED_PLATFORMS:
            if use_subdirectories:
                lib_path = self._target_lib_dir()
            else:
                lib_path = self._target_qairt_dir()

            if self._device.device_info.platform_type != DevicePlatformType.X86_64_LINUX:
                env.environment_variables["LD_LIBRARY_PATH"] = lib_path
                env.environment_variables["ADSP_LIBRARY_PATH"] = lib_path
                env.shell = True

        return env

    def _cleanup_device_artifacts(self):
        """Remove artifacts from the device."""
        if self._target_root_exists:
            self._check_device_return(
                self._device.remove(self._target_root), "Failed to remove artifacts from device\t"
            )
            self._target_root_exists = False
            self._config_artifacts_loaded = False
            self._sdk_artifacts_loaded = False
