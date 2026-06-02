# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

param (
    [switch]$h = $false,
    [string]$arch
)

Function _usage {
    $COMMAND_NAME = split-path $MyInvocation.PSCommandPath -Leaf
    Write-Output "Usage: ./$COMMAND_NAME [-h]"
    Write-Output ""
    Write-Output "Set up required environment variables for SDK deployment on host"
    Write-Output ""
    Write-Output "Parameters:"
    Write-Output "-h        Help"
    Write-Output "-arch     architecture(X86_64,ARM64)"
    Write-Output ""
    Write-Output "Example:"
    Write-Output "./$COMMAND_NAME"
    Write-Output ""
}

Function detect_cpu_instruction_set {
    $HwInfo = Get-CimInstance -ClassName Win32_Processor | Select-Object -Property Architecture
    if ($HwInfo.Architecture -eq 9 ) { # x86_64
        return "X86_64"
    }
    elseif ($HwInfo.Architecture -eq 12) { # ARM64
        return "ARM64"
    }
    else {
        Write-Error "Unsupported Architecture"
    }
}

Function setup_python_lib_path {
    param(
        [string]$QAIRT_SDK_ROOT
    )

    $PythonLib = $QAIRT_SDK_ROOT + "\lib\python"

    if ($env:PYTHONPATH) {
        $env:PYTHONPATH = ($env:PYTHONPATH.Split(';') | Where-Object -FilterScript {$_ -notlike "$QAIRT_SDK_ROOT*"}) -join ';'
        if ($env:PYTHONPATH) {
            $env:PYTHONPATH = $env:PYTHONPATH + [IO.Path]::PathSeparator + $PythonLib
        }
        else {
            $env:PYTHONPATH = $PythonLib
        }
    }
    else {
        $env:PYTHONPATH = $PythonLib
    }
}

Function setup_bin_lib_path {
    param (
        [string]$QAIRT_SDK_ROOT,
        [string]$InstructionSet
    )

    if ($InstructionSet -eq "X86_64") {
        $BinPath = $QAIRT_SDK_ROOT + "\bin\x86_64-windows-msvc"
        $LibPath = $QAIRT_SDK_ROOT + "\lib\x86_64-windows-msvc"
    }
    elseif ($InstructionSet -eq "ARM64") {
        $BinPath = $QAIRT_SDK_ROOT + "\bin\aarch64-windows-msvc"
        $LibPath = $QAIRT_SDK_ROOT + "\lib\aarch64-windows-msvc"
    }
    else {
        Write-Error "Unknown Architecture"
        Exit
    }

    if (-not ($BinPath | Test-Path) -or -not ($LibPath | Test-Path)) {
        Write-Error "Unsupported Architecture"
        Exit
    }

    $env:Path = ($env:Path.Split(';') | Where-Object -FilterScript {$_ -notlike "$QAIRT_SDK_ROOT*"}) -join ';'

    $env:Path = $BinPath + [IO.Path]::PathSeparator + $env:Path
    $env:Path = $LibPath + [IO.Path]::PathSeparator + $env:Path
}

Function setup_qairt_sdk {
     param (
        [string]$InstructionSet
    )

    $env:QAIRT_SDK_ROOT = (get-item $PSScriptRoot).parent.FullName
    Set-Variable -Name "QAIRT_SDK_ROOT" -Value $env:QAIRT_SDK_ROOT -Scope Global
    Write-Output "[INFO] QAIRT_SDK_ROOT: $env:QAIRT_SDK_ROOT"

    if (Test-Path "$env:QAIRT_SDK_ROOT\include\QNN") {
        Set-Variable -Name "QNN_SDK_ROOT" -Value $env:QAIRT_SDK_ROOT -Scope Global
        $env:QNN_SDK_ROOT = $QNN_SDK_ROOT
    }
    else {
      Write-Output "[WARNING] QNN API files are missing from QAIRT SDK. Please verify the integrity of your download."
    }

    if (Test-Path "$env:QAIRT_SDK_ROOT\include\SNPE") {
        Set-Variable -Name "SNPE_ROOT" -Value $env:QAIRT_SDK_ROOT -Scope Global
        $env:SNPE_ROOT = $SNPE_ROOT
    }

    setup_bin_lib_path $env:QAIRT_SDK_ROOT $InstructionSet
    setup_python_lib_path $env:QAIRT_SDK_ROOT
}

if ($h) {
    _usage
    Exit
}

if ($arch) {
    if (($arch.ToUpper() -eq "X86_64") -or ($arch.ToUpper() -eq "ARM64")) {
        setup_qairt_sdk $arch.ToUpper()
    }
    else {
        Write-Error "Unsupported Architecture"
        Exit
    }
}
else {
    $arch = detect_cpu_instruction_set
    setup_qairt_sdk $arch
}

if ((Test-Path "$env:AISW_SDK_ROOT\include\QNN") -and (Test-Path "$env:AISW_SDK_ROOT\include\SNPE")) {
    Write-Output echo "[WARN] QNN_SDK_ROOT/SNPE_ROOT set to QAIRT_SDK_ROOT for backwards compatibility and will be deprecated in a future release."
}
elseif (Test-Path "$env:AISW_SDK_ROOT\include\QNN") {
    Write-Output echo "[WARN] QNN_SDK_ROOT set to QAIRT_SDK_ROOT for backwards compatibility and will be deprecated in a future release."
}

Write-Output "[INFO] QAIRT SDK environment setup complete"
