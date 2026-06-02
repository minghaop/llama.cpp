# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""File contains a class which helps to convert a dictionary containing name to tensor mapping to on-disk weights"""

import tempfile
import json
import numpy as np
import pathlib as pl
import shutil

class WeightLoader:
    """
    Creates an on-disk weights file and provides dictionary like interface.
    """
    def __init__(self, data, cache_dir:pl.Path = None, clean_start=True):

        self.is_tmp = False
        self._temp_dir_obj = None
        self._allow_update = False

        # In case the cache_dir is provided the data directory is persistent,
        # else create a temp directory and delete it once the object gets killed.
        if cache_dir is None:
            self._temp_dir_obj = tempfile.TemporaryDirectory()
            cache_dir = self._temp_dir_obj.name
            self.is_tmp = True

        self.cache_dir = cache_dir
        self.root = pl.Path(self.cache_dir)
        if clean_start and self.root.exists():
            self._safe_rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

        # Store the data keys for dictionary-like access
        self._data_keys = list(data.keys())

        # Save numpy arrays to files
        for key, value in data.items():
            np.save(self.root / f"{key}.npy", value)

        # Save metadata, used later to load the data from the directory.
        with open(self.root / "meta.json", "w") as f:
            json.dump({"names": self._data_keys}, f)

    @classmethod
    def from_directory(cls, directory_path: pl.Path):
        """Loads the data from a given directory and returns the WeightLoader object """
        directory_path = pl.Path(directory_path)
        if not (directory_path / "meta.json").is_file():
            raise ValueError("Invalid Directory Found. Not weights to load!")

        with open(directory_path / "meta.json", "r") as f:
            data_keys = json.load(f)
            if "names" not in data_keys:
                raise ValueError("Invalid meta information found :( ")

        data_obj = cls.__new__(cls)
        data_obj.is_tmp = False
        data_obj._temp_dir_obj = None
        data_obj.cache_dir = directory_path
        data_obj.root = directory_path
        data_obj._data_keys = data_keys["names"]
        data_obj._allow_update = False
        return data_obj

    @property
    def is_updatable(self):
        return self._allow_update

    def set_updatable(self, alternate_save_path: pl.Path = None):
        self._allow_update  = True

        if alternate_save_path:
            alternate_save_path = pl.Path(alternate_save_path)

            # Copy the content from current root to alternate_path
            self._transfer_content(self.root, self._data_keys, alternate_save_path)

            # Update the root, so that the new data will be saved at the new location
            self.root = alternate_save_path


    def move_to(self, save_path):
        save_path = pl.Path(save_path)

        if self.root == save_path:
            return

        self._transfer_content(self.root, self._data_keys, save_path, keep_original=False)
        self.root = save_path

    @staticmethod
    def _transfer_content(source, keys, destination, keep_original:bool = True):
        # Make sure the destination exists
        destination.mkdir(parents=True, exist_ok=True)

        copy_fn = shutil.copy if keep_original else shutil.move

        # First transfer the meta.json
        copy_fn(source / "meta.json", destination / "meta.json")

        # Transfer all the data files
        for key in keys:
            src = source / f"{key}.npy"
            dst = destination / f"{key}.npy"
            copy_fn(src, dst)


    def __del__(self):
        """Destructor that deletes generated files only if temp directory was created"""
        if self.is_tmp and hasattr(self, 'root') and self.root.exists():
            try:
                # Clean up temporary directory and all its contents
                if self._temp_dir_obj is not None:
                    self._temp_dir_obj.cleanup()
                else:
                    # Fallback cleanup if temp_dir_obj is not available
                    shutil.rmtree(self.root, ignore_errors=True)
            except Exception:
                # Ignore cleanup errors to prevent issues during destruction
                pass

    def __getitem__(self, key):
        """Enable dictionary-like access to load data"""
        if key not in self._data_keys:
            raise KeyError(f"Key '{key}' not found in weight loader")

        npy_path = self.root / f"{key}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Data file for key '{key}' not found")

        return np.load(npy_path, allow_pickle=False)

    def __setitem__(self, key, value):
        if not self._allow_update:
            raise ValueError("Update not allowed. In case the update is necessary, "
                             "first enable the updates by calling `set_updatable`")
        if key not in self._data_keys:
            raise KeyError(f"Unknown key {key}, New key addition is not supported!")

        np.save(self.root / f"{key}.npy", value)


    def __contains__(self, key):
        """Enable 'in' operator for checking if key exists"""
        return key in self._data_keys

    def items(self):
        """Return key-value pairs (generator to avoid loading all data at once)"""
        for key in self._data_keys:
            yield key, self[key]

    def __len__(self):
        """Return number of stored items"""
        return len(self._data_keys)

    def keys(self):
        return list(self._data_keys)

    def values(self):
        for k in self._data_keys:
            yield self[k]

    @staticmethod
    def _safe_rmtree(p: pl.Path):
        # Safety check to avoid removing sensitive directories
        if not p.exists():
            return
        if p == p.anchor:
            raise RuntimeError(f"Refusing to remove root directory: {p}")
        shutil.rmtree(p, ignore_errors=True)
