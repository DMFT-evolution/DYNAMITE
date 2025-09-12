#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace h5rt {

// Minimal HDF5 C API type aliases
using hid_t  = long long;  // matches HDF5 hid_t width on 64-bit
using herr_t = int;
using hsize_t = unsigned long long;

// Loader API
bool try_load();      // attempt to dlopen libhdf5; safe to call multiple times
bool available();     // true if libhdf5 loaded successfully
void unload();        // optional: dlclose

// High-level helpers used by io_utils

// Open/close files
hid_t open_file_readonly(const char* path);      // H5Fopen(..., H5F_ACC_RDONLY, H5P_DEFAULT)
hid_t create_file_trunc(const char* path);       // H5Fcreate(..., H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)
void  close_file(hid_t file);

// Attributes (scalars)
bool read_attr_double(hid_t file, const char* name, double& out);
bool read_attr_int(hid_t file, const char* name, int& out);
bool write_attr_double(hid_t file, const char* name, double value);
bool write_attr_int(hid_t file, const char* name, int value);
bool write_attr_string(hid_t file, const char* name, const std::string& value);

// 1D dataset read/write of doubles
bool read_dataset_1d_double(hid_t file, const char* name, std::vector<double>& out);
bool write_dataset_1d_double(hid_t file, const char* name, const double* data, size_t n,
                             int compression_level = 6, size_t chunk = 4096);

// Utility
size_t dataset_length(hid_t file, const char* name); // returns 0 if not found

} // namespace h5rt
