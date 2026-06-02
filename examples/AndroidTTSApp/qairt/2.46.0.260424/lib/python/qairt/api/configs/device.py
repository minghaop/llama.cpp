# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import ipaddress
import logging
import re
from typing import Any, List, Optional, Union

# TODO: Handle qti imports in a common location
from qti.aisw.tools.core.modules.api.definitions import AISWBaseModel
from qti.aisw.tools.core.utilities.devices.android.android_device import AndroidDevice
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    ConnectionType,
    DeviceCredentials,
    DeviceEnvironmentContext,
    DeviceIdentifier,
    DeviceInfo,
    DevicePlatformType,
    RemoteDeviceIdentifier,
    RemoteDeviceInfo,
    SocDetails,
)
from qti.aisw.tools.core.utilities.devices.api.device_factory import DeviceFactory
from qti.aisw.tools.core.utilities.devices.api.device_interface import DeviceInterface

# The regex below matches
# An ADB id in the format device_id[@<host>][:<port>] where host and port are optional
android_identifier_regex = re.compile(r"^([a-zA-Z0-9:_-]+)(?:@([a-zA-Z0-9.\-]+)(?::(\d+))?)?$")

# A login identifier in the format <username>[:password]@hostname[:port] where password and port are optional
login_identifier_regex = re.compile(r"^([a-zA-Z0-9._-]+)(?::([^@]+))?@([a-zA-Z0-9.\-]+)(?::(\d+))?$")


def split_adb_target(adb_identifier: str) -> tuple[str, Optional[str], Optional[int]]:
    """Split an ADB identifier into its components.

    Args:
        adb_identifier (str): The ADB identifier to split.

    Returns:
        tuple: A tuple containing the serial ID, hostname, and port.

    Raises:
        ValueError: If the ADB identifier is invalid.
    """
    match = re.match(android_identifier_regex, adb_identifier)
    if not match:
        raise ValueError(f"Invalid ADB identifier: {adb_identifier}")
    serial_id, hostname, port = match.groups()
    return serial_id, hostname if hostname else None, int(port) if port else None


def split_login_identifier(
    login_identifier: str,
) -> tuple[str | Any, str | Any, Optional[str], Optional[int]]:
    """Split a login identifier into its components.

    Args:
        login_identifier (str): The login identifier to split.

    Returns:
        tuple: A tuple containing the username, password, hostname, and port.

    Raises:
        ValueError: If the login identifier is invalid.
    """
    match = re.match(login_identifier_regex, login_identifier)
    if not match:
        raise ValueError(f"Invalid login identifier: {login_identifier}")
    username, password, hostname, port = match.groups()
    return username, password if password else "", hostname, int(port) if port else None


class Device(AISWBaseModel):
    """
    An object that captures information about the intended device.

    Supported Platform types are:
         - DevicePlatformType.ANDROID
         - DevicePlatformType.LINUX_EMBEDDED
         - DevicePlatformType.QNX
         - DevicePlatformType.WOS
         - DevicePlatformType.X86_64_LINUX
         - DevicePlatformType.X86_64_WINDOWS_MSVC

    For remote connections, additional information such as hostname or ip, and credentials
    may be required.

    This could be passed through the identifier using a pattern.

    For Android and linux embedded devices, the identifier pattern is adb_serial_id[@<host>][:<port>]
    where host and port are optional.

    For QNX, WoS and X86_64 Windows devices, the identifier pattern is <username>[:password]@hostname[:port]
    where password and port are optional.

    Examples:

    .. code-block:: python

        # For android or linux embedded devices, port is optional
        android_device = Device(type=DevicePlatformType.ANDROID, identifier="a1234bc")
        android_device_remote = Device(type=DevicePlatformType.ANDROID, identifier="a1234bc@hostname:5555")

        # For QNX devices, port is optional
        qnx_device =  Devices(type=DevicePlatformType.QNX, identifier=username@hostname:port)

    """

    type: DevicePlatformType
    """The type of device platform to be used"""
    identifier: Optional[Union[str, RemoteDeviceIdentifier]] = None
    """The identifier of the device.Defaults to None."""
    credentials: Optional[DeviceCredentials] = None
    """The credentials for the device. Defaults to None."""

    _info: Optional[Union[RemoteDeviceInfo, DeviceInfo]] = None

    def model_post_init(self, __context: Any) -> None:
        if isinstance(self.identifier, str):
            if self.type in [DevicePlatformType.ANDROID, DevicePlatformType.LINUX_EMBEDDED]:
                # attempt to split the identifier if it is a raw str
                serial_id, hostname, port = split_adb_target(self.identifier)
                if not serial_id:
                    raise ValueError(f"Invalid ADB identifier: {self.identifier}")
                self.identifier = RemoteDeviceIdentifier(serial_id=serial_id, hostname=hostname, port=port)
            elif self.type in [
                DevicePlatformType.QNX,
                DevicePlatformType.X86_64_WINDOWS_MSVC,
                DevicePlatformType.WOS,
            ]:
                # attempt to split the identifier if it is a raw str
                username, password, hostname_or_ip, port = split_login_identifier(str(self.identifier))
                if not (username and hostname_or_ip):
                    raise ValueError(f"Invalid login identifier: {self.identifier}")
                self.credentials = DeviceCredentials(username=username, password=password)

                # check if the identifier is a hostname or ip address
                try:
                    ipaddress.ip_address(hostname_or_ip)
                    self.identifier = RemoteDeviceIdentifier(ip_addr=hostname_or_ip, port=port)
                except ValueError:
                    self.identifier = RemoteDeviceIdentifier(hostname=hostname_or_ip, port=port)

    @property
    def info(self) -> DeviceInfo:
        if self._info:
            pass
        elif self.identifier and isinstance(self.identifier, RemoteDeviceIdentifier):
            self._info = RemoteDeviceInfo(
                platform_type=self.type, identifier=self.identifier, credentials=self.credentials
            )
        else:
            self._info = DeviceInfo(platform_type=self.type, identifier=self.identifier)

        return self._info

    def get_chipset(self) -> str:
        """Returns the chipset associated with this device. Note that only
        Android and OE-Linux devices are supported."""
        if self.type in (DevicePlatformType.ANDROID, DevicePlatformType.LINUX_EMBEDDED):
            device_factory_instance = DeviceFactory.create_device(self.info)
            assert isinstance(device_factory_instance, AndroidDevice)
            chipset = device_factory_instance.get_soc_name()
            if not chipset:
                print(f"Could not resolve chipset for android device. Chipset set to UNKNOWN")
            return chipset
        else:
            print(f"Could not resolve chipset for device of type: {self.type} ")
            return ""

    def get_log(self, **kwargs) -> bytes | str | None:
        """
        Get the entire device log, filter by date range, and/or filter by a string expression.

        Args:
            **kwargs: Keyword arguments.

                - date_range (Optional[Dict[str, datetime|str]): The date range of the device log.
                  A date range is defined by a start and end datetime
                  in the format {"start": datetime, "end": datetime}
                  where datetime is a datetime object or str.
                - log_filter (Optional[str]): A string expression to filter logs (passed to re.compile).
                  Defaults to a case-insensitive match of the words: qnn or genie.
                - max_lines (Optional[int]):  (Android Only) Truncate formatted output to last N lines
                  (after filtering).
                - use_device_time_hint (Optional[bool]): (Android only) If True, attempt to use '-t' date_range.start
                  as an argument to 'adb logcat' to obtain logs since a given
                  timestamp.

        Returns:
            bytes: The device log content
        """
        if self.type in [
            DevicePlatformType.ANDROID,
            DevicePlatformType.LINUX_EMBEDDED,
            DevicePlatformType.QNX,
        ]:
            log_filter = kwargs.get("log_filter", None)
            if not log_filter:
                kwargs["log_filter"] = r"(?i)^(?=.*(qnn|genie)).+$"
            device_factory_instance = DeviceFactory.create_device(self.info)
            return device_factory_instance.get_device_log(**kwargs)
        else:
            raise NotImplementedError(f"Log retrieval not implemented for device type: {self.type}")

    def __str__(self):
        return f"{str(self.model_dump(exclude_none=True, exclude_unset=True, exclude={'credentials'}))}"


def soc_details_from_str(specs_str):
    return get_soc_details(specs_str)[0]


def populate_soc_details_from_factory(soc_detail: SocDetails, backend: str = "HTP"):
    """Populates soc detail with additional information given a backend. This function
    will override any preset values."""

    if not soc_detail.chipset:
        print(f"Could not determine soc details without chipset")
        return False

    try:
        soc_details_from_factory = DeviceFactory.get_device_soc_details(backend, soc_detail.chipset)
        assert soc_details_from_factory is not None
        soc_detail.dsp_arch = soc_details_from_factory.dsp_arch
        soc_detail.model = soc_details_from_factory.model
        soc_detail.num_of_hvx_threads = soc_details_from_factory.num_of_hvx_threads
        soc_detail.vtcm_size_in_mb = soc_details_from_factory.vtcm_size_in_mb
        soc_detail.supports_fp16 = soc_details_from_factory.supports_fp16
    except Exception as e:
        print(f"Could not determine soc details due to: {e}")
        return False
    return True


def get_soc_details(specs_str: str) -> List[SocDetails]:
    """Transforms a spec str into a device spec object

    Matches one or more instances of either:
        1. chipset:abc-123 e.x chipset:SM8550
        2. dsp_arch:v123;soc_model:123|456 e.x dsp_arch:v73;soc_model:43|50
        3. chipset:abc-123;dsp_arch:v123;soc_model:123|456 e.x chipset:SM8550;dsp_arch:v73;soc_model:43|50
        4. Any of the above with an optional ;cores:N suffix e.x chipset:SM8550;cores:4

    Returns:
        A list of soc detail objects

    Raises:
        ValueError: If the specs string is not valid or no matches were found
    """

    # Strip and capture cores entry before existing parsing
    cores_match = re.search(r"cores:(\d+)", specs_str)
    if cores_match:
        specs_str = re.sub(r";?cores:\d+", "", specs_str)

    pattern = r"(chipset:[\w-]+|dsp_arch:[\w-]+;soc_model:\d+(\|\d+)*)"

    # Find all matches in the input string
    matches = re.findall(pattern, specs_str)

    # Create a list of SocDetails instances from the matches
    soc_details_default = lambda: SocDetails(
        chipset="", model="", dsp_arch=0, num_of_hvx_threads=0, vtcm_size_in_mb=0, supports_fp16=False
    )
    soc_details = soc_details_default()

    if not matches and specs_str:
        raise ValueError("Input does not match expected format")

    soc_details_list = []

    # To illustrate the code below consider the example:
    # chipset:SM8550;dsp_arch:v73;soc_model:60
    # The matches are: [("chipset:SM8550", ""), (";dsp_arch:v73;soc_model:69", '')]
    # The first pass through adds chipset to a partially formed soc detail.
    # The second pass populates the same soc detail with a dsp arch and soc model.
    for match in matches:
        entry = match[0]

        if entry.startswith("chipset:"):
            # if chipset or dsp arch has already been found,
            # then create a new soc detail
            if soc_details.chipset or soc_details.dsp_arch:
                soc_details_list.append(soc_details)
                soc_details = soc_details_default()

            soc_details.chipset = entry.split(":")[1]

        elif entry.startswith("dsp_arch:"):
            dsp_arch_part, soc_model_part = entry.split(";")
            dsp_arch = dsp_arch_part.split(":")[1]
            soc_model = soc_model_part.split(":")[1]

            # TODO: Remove when dsp arch is a str by default
            idx = dsp_arch.find("v")
            if idx != -1:
                dsp_arch = dsp_arch[idx + 1 :]

            soc_details.dsp_arch = int(dsp_arch)
            soc_details.model = soc_model

            if not soc_details.model:
                raise ValueError("Soc model should be provided along with dsp arch")

            if not soc_details.chipset:
                # if chipset is not populated
                soc_details_list.append(soc_details)
                soc_details = soc_details_default()

    # add final soc details to list
    if soc_details.chipset or soc_details.dsp_arch:
        soc_details_list.append(soc_details)

    if cores_match:
        for sd in soc_details_list:
            sd.num_cores = int(cores_match.group(1))

    return soc_details_list


def log_device_output(
    device: Optional[Union[Device, DeviceInterface]], logger: logging.Logger, **device_log_kwargs
) -> None:
    """Log device output for debugging purposes."""
    try:
        if device is not None and isinstance(device, DeviceInterface):
            assert device.device_info is not None
            device = Device(identifier=device.device_info.identifier, type=device.device_info.platform_type)

        device_log_kwargs["max_lines"] = device_log_kwargs.get("max_lines", 500)
        full_log = device.get_log(**device_log_kwargs) if device else None
        if full_log:
            logger.debug(f"-- Full Device Log (first {device_log_kwargs['max_lines']} lines) --")
            logger.debug(
                full_log.decode("utf-8", errors="replace") if isinstance(full_log, bytes) else full_log
            )
        else:
            logger.debug("<no log content returned>. See device.get_log() for details if device is set.")
    except Exception as e:
        logger.debug(f"Failed to retrieve full device log: {e}")
