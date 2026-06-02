#!/bin/bash
#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

function usage() {
  echo "Usage: $0 [-n|--dry_run, -h|--help]"
  echo "  -h, --help      show this help message"
  echo "  -n, --dry_run   perform the check but dont install"
}

function parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) usage; exit 0 ;;
      -n|--dry-run) dry_run=true; shift ;;
      *) echo "Unknown arg: $1"; usage; exit 1 ;;
    esac
  done
}

function finish_setup() {
  exit_status=$?

  set +e

  # return to dir where script was executed from if not already there
  popd &> /dev/null || exit

  # clean up script functions and variables setup from environment
  unset -f _usage
  unset -f param
  unset -f install_pkg
  unset -f finish_setup
  unset -f verify_pkg_installed
  unset -f setup_op_package_dependencies
  unset -f setup_accuracy_debugger_dependencies
  unset -f setup_clang
  unset -f setup_flatbuffers-compiler
  unset -f setup_libflatbuffers-dev
  unset -f setup_rename

  # remove trapping from shell
  trap '' EXIT RETURN

  if [ ${exit_status} -ne 0 ]; then
    echo "[ERROR] Linux dependency check: Failed."
    return 1
  fi

  return 0
}

function verify_pkg_installed() {
  echo "$(dpkg-query -W --showformat='${Status}\n' "$1"|grep "install ok installed")"
}

function install_pkg() {
  local pkg="${1}"
  local deps_group="${2}"
  if ${dry_run}; then
    need_to_install+=("${pkg}")
  else
    apt-get update
    apt-get install -y "${pkg}"
    retVal=$?
    if [[ $retVal -ne 0 ]]; then
      echo "ERROR: Failed to install required package: ${pkg} for ${deps_group}"
      exit 1
    fi
  fi
}

function setup_op_package_dependencies() {
  pkgs_to_check=('libncurses5')
  j=0
  while [ $j -lt ${#pkgs_to_check[*]} ]; do
    echo "Checking for ${pkgs_to_check[$j]}"
    install_status=$(verify_pkg_installed "${pkgs_to_check[$j]}")
    if [ "$install_status" == "" ]; then
      install_pkg "${pkgs_to_check[$j]}" "op package generation"
    fi
    j=$(( j + 1 ));
  done
}

function setup_accuracy_debugger_dependencies() {
  pkgs_to_check=('libgl1')
  j=0
  while [ $j -lt ${#pkgs_to_check[*]} ]; do
    echo "Checking for ${pkgs_to_check[$j]}"
    install_status=$(verify_pkg_installed ${pkgs_to_check[$j]})
    if [ "$install_status" == "" ]; then
      install_pkg ${pkgs_to_check[$j]} "accuracy debugger"
    fi
    j=$(( $j +1));
  done
}

function setup_clang() {
  if [ $(lsb_release -sr) == 20.04 ]; then
    pkgs_to_check=('clang-9' 'libc++-9-dev')
  else
    pkgs_to_check=('clang' 'libc++-dev' 'libc++abi-dev')
  fi
  j=0
  while [ $j -lt ${#pkgs_to_check[*]} ]; do
    echo "Checking for ${pkgs_to_check[$j]}"
    install_status=$(verify_pkg_installed ${pkgs_to_check[$j]})
    if [ "$install_status" == "" ]; then
      install_pkg ${pkgs_to_check[$j]} "clang"
    fi
    j=$(( $j +1));
  done
}

function setup_flatbuffers-compiler() {
  apt-get update
  pkgs_to_check=('software-properties-common')
  j=0
  while [ $j -lt ${#pkgs_to_check[*]} ]; do
    install_status=$(verify_pkg_installed "${pkgs_to_check[$j]}")
    if [ "$install_status" == "" ]; then
      apt-get install -y "${pkgs_to_check[$j]}"
      if [[ $? -ne 0 ]]; then
        echo "ERROR: Failed to install required packages for flatbuffers-compiler"
        exit 1
      fi
    fi
    j=$(( j + 1 ));
  done

  add-apt-repository 'deb http://cz.archive.ubuntu.com/ubuntu focal main universe'
  apt-get update
  apt-get install -y 'flatbuffers-compiler'
  echo "Setup Complete. flatc installed at /usr/bin/flatc"
}

function setup_libflatbuffers-dev() {
  apt-get update
  pkgs_to_check=('software-properties-common')
  j=0
  while [ $j -lt ${#pkgs_to_check[*]} ]; do
    install_status=$(verify_pkg_installed "${pkgs_to_check[$j]}")
    if [ "$install_status" == "" ]; then
      apt-get install -y "${pkgs_to_check[$j]}"
      retVal=$?
      if [[ $retVal -ne 0 ]]; then
        echo "ERROR: Failed to install required packages for libflatbuffers-dev"
        exit 1
      fi
    fi
    j=$(( j + 1 ));
  done

  add-apt-repository 'deb http://cz.archive.ubuntu.com/ubuntu focal main universe'
  apt-get update
  apt-get install -y 'libflatbuffers-dev'
  echo "Setup Complete. libflatbuffers-dev installed."
}

function setup_rename() {
  apt-get update
  pkgs_to_check=('software-properties-common')
  j=0
  while [ $j -lt ${#pkgs_to_check[*]} ]; do
    install_status=$(verify_pkg_installed "${pkgs_to_check[$j]}")
    if [ "$install_status" == "" ]; then
      apt-get install -y "${pkgs_to_check[$j]}"
      retVal=$?
      if [[ $retVal -ne 0 ]]; then
        echo "ERROR: Failed to install required packages for rename"
        exit 1
      fi
    fi
    j=$(( j +1 ));
  done

  add-apt-repository 'deb http://cz.archive.ubuntu.com/ubuntu focal main universe'
  apt-get update
  apt-get install -y 'rename'
  echo "Setup Complete. rename installed at /usr/bin/rename"
}

# setup clean up of files for on any exit.
trap finish_setup EXIT RETURN

dry_run=false
parse_args "$@" || exit 1

# Check if Ubuntu version is not 20.04 or 22.04
version=$(lsb_release -rs)
if [ "$version" != "20.04" ] && [ "$version" != "22.04" ]; then
  echo "Warning: Ubuntu Versions 20.04 and 22.04 are supported. Detected Ubuntu version $version"
fi

# Linux Package dependencies that are needed for QNN SDK
needed_depends=()
needed_depends+=('flatbuffers-compiler')
needed_depends+=('libflatbuffers-dev')
needed_depends+=('rename')

#Unmet dependencies
need_to_install=()
i=0

setup_op_package_dependencies
setup_accuracy_debugger_dependencies
setup_clang

while [ $i -lt ${#needed_depends[*]} ]; do
  echo "Checking for ${needed_depends[$i]}"
  case "${needed_depends[$i]}" in
    flatbuffers-compiler )
      PKG_INSTALLED=$(verify_pkg_installed "${needed_depends[$i]}")
      if [ "$PKG_INSTALLED" == "" ]; then
        if [ -d /usr/bin/flatc ]; then
          case ":$PATH:" in
            *:/usr/bin/flatc:*) echo "flatc Found: /usr/bin/flatc";;
            *) export PATH=/usr/bin/flatc:$PATH
               echo "flatc Found: /usr/bin/flatc. Added to PATH variable";;
          esac
        else
          echo "No pre-installed ${needed_depends[$i]} is found!!"
          need_to_install+=("${needed_depends[$i]}")
        fi
      fi
      ;;
    libflatbuffers-dev )
      PKG_INSTALLED=$(verify_pkg_installed "${needed_depends[$i]}")
      if [ "$PKG_INSTALLED" == "" ]; then
        if [ -d /usr/include/flatbuffers ]; then
          echo "libfatbuffers-dev found at /usr/include/flatbuffers"
        else
          echo "No pre-installed ${needed_depends[$i]} is found!!"
          need_to_install+=("${needed_depends[$i]}")
        fi
      fi
      ;;
    rename )
      PKG_INSTALLED=$(verify_pkg_installed "${needed_depends[$i]}")
      if [ "$PKG_INSTALLED" == "" ]; then
        if [ -d /usr/bin/rename ]; then
          case ":$PATH:" in
            *:/usr/bin/rename:*) echo "rename Found: /usr/bin/rename";;
            *) export PATH=/usr/bin/rename:$PATH
               echo "rename found at /usr/bin/rename"
          esac
        else
          echo "No pre-installed ${needed_depends[$i]} is found!!"
          need_to_install+=("${needed_depends[$i]}")
        fi
      fi
      ;;
  esac
  i=$(( i + 1 ));
done


echo "============================================================="
if [ ${#need_to_install[*]} -ne 0 ]; then
  if ${dry_run}; then
    echo "Missing required packages: ${need_to_install[*]}"
    exit 1
  else
    i=0
      echo "Installing Missing Packages"
      echo "--------------------------------------------------------------"
      while [ $i -lt ${#need_to_install[*]} ]; do
        echo "Setting up Package: ${need_to_install[$i]}"
        case "${need_to_install[$i]}" in
          flatbuffers-compiler )
            setup_flatbuffers-compiler
            ;;
          libflatbuffers-dev )
            setup_libflatbuffers-dev
            ;;
          rename )
            setup_rename
            ;;
        esac
        i=$(( $i +1));
        echo "--------------------------------------------------------------"
      done
  fi
else
  echo "All Dependency Packages Found"
fi
echo "Done!!"
