#include "io/h5_runtime.hpp"
#include <dlfcn.h>
#include <mutex>
#include <vector>
#include <cstring>

namespace h5rt {

namespace {
void* handle = nullptr;
std::once_flag once;

template <typename T>
T sym(const char* n) { return reinterpret_cast<T>(dlsym(handle, n)); }

using hid_t = h5rt::hid_t;
using herr_t = h5rt::herr_t;
using hsize_t = h5rt::hsize_t;

// Function pointer typedefs for needed HDF5 C API calls
using p_H5open = herr_t (*)();
using p_H5close = herr_t (*)();
using p_H5Fopen = hid_t (*)(const char*, unsigned, hid_t);
using p_H5Fcreate = hid_t (*)(const char*, unsigned, hid_t, hid_t);
using p_H5Fclose = herr_t (*)(hid_t);
using p_H5Aopen_by_name = hid_t (*)(hid_t, const char*, const char*, hid_t, hid_t);
using p_H5Acreate2 = hid_t (*)(hid_t, const char*, hid_t, hid_t, hid_t, hid_t);
using p_H5Aread = herr_t (*)(hid_t, hid_t, void*);
using p_H5Awrite = herr_t (*)(hid_t, hid_t, const void*);
using p_H5Aclose = herr_t (*)(hid_t);
using p_H5Dopen2 = hid_t (*)(hid_t, const char*, hid_t);
using p_H5Dcreate2 = hid_t (*)(hid_t, const char*, hid_t, hid_t, hid_t, hid_t, hid_t);
using p_H5Dread = herr_t (*)(hid_t, hid_t, hid_t, hid_t, hid_t, void*);
using p_H5Dwrite = herr_t (*)(hid_t, hid_t, hid_t, hid_t, hid_t, const void*);
using p_H5Dclose = herr_t (*)(hid_t);
using p_H5Dget_space = hid_t (*)(hid_t);
using p_H5Screate_simple = hid_t (*)(int, const hsize_t*, const hsize_t*);
using p_H5Sclose = herr_t (*)(hid_t);
using p_H5Sget_simple_extent_dims = int (*)(hid_t, hsize_t*, hsize_t*);
using p_H5Pcreate = hid_t (*)(hid_t);
using p_H5Pset_deflate = herr_t (*)(hid_t, unsigned);
using p_H5Pset_chunk = herr_t (*)(hid_t, int, const hsize_t*);
using p_H5Pclose = herr_t (*)(hid_t);

// Constants (copied values from HDF5 headers)
constexpr unsigned H5F_ACC_RDONLY = 0x0000u;
constexpr unsigned H5F_ACC_TRUNC  = 0x0002u;
constexpr hid_t H5P_DEFAULT = 0;

// Native types (H5T_NATIVE_DOUBLE, H5T_NATIVE_INT)
// Using indirect lookup from dynamic lib; declare as hid_t constants that we resolve.
hid_t H5T_NATIVE_DOUBLE = -1;
hid_t H5T_NATIVE_INT = -1;

// Dynamic function pointers
p_H5open s_H5open = nullptr; p_H5close s_H5close = nullptr;
p_H5Fopen s_H5Fopen = nullptr; p_H5Fcreate s_H5Fcreate = nullptr; p_H5Fclose s_H5Fclose = nullptr;
p_H5Aopen_by_name s_H5Aopen_by_name = nullptr; p_H5Acreate2 s_H5Acreate2 = nullptr; p_H5Aread s_H5Aread = nullptr; p_H5Awrite s_H5Awrite = nullptr; p_H5Aclose s_H5Aclose = nullptr;
p_H5Dopen2 s_H5Dopen2 = nullptr; p_H5Dcreate2 s_H5Dcreate2 = nullptr; p_H5Dread s_H5Dread = nullptr; p_H5Dwrite s_H5Dwrite = nullptr; p_H5Dclose s_H5Dclose = nullptr;
p_H5Dget_space s_H5Dget_space = nullptr;
p_H5Screate_simple s_H5Screate_simple = nullptr; p_H5Sclose s_H5Sclose = nullptr; p_H5Sget_simple_extent_dims s_H5Sget_simple_extent_dims = nullptr;
p_H5Pcreate s_H5Pcreate = nullptr; p_H5Pset_deflate s_H5Pset_deflate = nullptr; p_H5Pset_chunk s_H5Pset_chunk = nullptr; p_H5Pclose s_H5Pclose = nullptr;

bool do_load() {
  const char* candidates[] = {"libhdf5.so", "libhdf5_serial.so"};
  for (const char* name : candidates) {
    handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (handle) break;
  }
  if (!handle) return false;

  s_H5open = sym<p_H5open>("H5open");
  s_H5close = sym<p_H5close>("H5close");
  s_H5Fopen = sym<p_H5Fopen>("H5Fopen");
  s_H5Fcreate = sym<p_H5Fcreate>("H5Fcreate");
  s_H5Fclose = sym<p_H5Fclose>("H5Fclose");
  s_H5Aopen_by_name = sym<p_H5Aopen_by_name>("H5Aopen_by_name");
  s_H5Acreate2 = sym<p_H5Acreate2>("H5Acreate2");
  s_H5Aread = sym<p_H5Aread>("H5Aread");
  s_H5Awrite = sym<p_H5Awrite>("H5Awrite");
  s_H5Aclose = sym<p_H5Aclose>("H5Aclose");
  s_H5Dopen2 = sym<p_H5Dopen2>("H5Dopen2");
  s_H5Dcreate2 = sym<p_H5Dcreate2>("H5Dcreate2");
  s_H5Dread = sym<p_H5Dread>("H5Dread");
  s_H5Dwrite = sym<p_H5Dwrite>("H5Dwrite");
  s_H5Dclose = sym<p_H5Dclose>("H5Dclose");
  s_H5Dget_space = sym<p_H5Dget_space>("H5Dget_space");
  s_H5Screate_simple = sym<p_H5Screate_simple>("H5Screate_simple");
  s_H5Sclose = sym<p_H5Sclose>("H5Sclose");
  s_H5Sget_simple_extent_dims = sym<p_H5Sget_simple_extent_dims>("H5Sget_simple_extent_dims");
  s_H5Pcreate = sym<p_H5Pcreate>("H5Pcreate");
  s_H5Pset_deflate = sym<p_H5Pset_deflate>("H5Pset_deflate");
  s_H5Pset_chunk = sym<p_H5Pset_chunk>("H5Pset_chunk");
  s_H5Pclose = sym<p_H5Pclose>("H5Pclose");

  // Resolve native types constants exported by libhdf5
  H5T_NATIVE_DOUBLE = *reinterpret_cast<hid_t*>(dlsym(handle, "H5T_NATIVE_DOUBLE_g"));
  H5T_NATIVE_INT    = *reinterpret_cast<hid_t*>(dlsym(handle, "H5T_NATIVE_INT_g"));

  if (s_H5open) s_H5open();
  return true;
}
} // namespace

bool try_load() {
  std::call_once(once, []{ (void)do_load(); });
  return handle != nullptr;
}

bool available() {
  if (!handle) (void)try_load();
  return handle != nullptr;
}

void unload() {
  if (handle) {
    if (s_H5close) s_H5close();
    dlclose(handle);
    handle = nullptr;
  }
}

hid_t open_file_readonly(const char* path) {
  if (!available() || !s_H5Fopen) return -1;
  return s_H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
}

hid_t create_file_trunc(const char* path) {
  if (!available() || !s_H5Fcreate) return -1;
  return s_H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
}

void close_file(hid_t file) {
  if (available() && s_H5Fclose && file >= 0) (void)s_H5Fclose(file);
}

bool read_attr_double(hid_t file, const char* name, double& out) {
  if (!available() || !s_H5Aopen_by_name || !s_H5Aread) return false;
  hid_t attr = s_H5Aopen_by_name(file, ".", name, 0, 0);
  if (attr < 0) return false;
  bool ok = (s_H5Aread(attr, H5T_NATIVE_DOUBLE, &out) >= 0);
  s_H5Aclose(attr);
  return ok;
}

bool read_attr_int(hid_t file, const char* name, int& out) {
  if (!available() || !s_H5Aopen_by_name || !s_H5Aread) return false;
  hid_t attr = s_H5Aopen_by_name(file, ".", name, 0, 0);
  if (attr < 0) return false;
  bool ok = (s_H5Aread(attr, H5T_NATIVE_INT, &out) >= 0);
  s_H5Aclose(attr);
  return ok;
}

bool write_attr_double(hid_t file, const char* name, double value) {
  if (!available() || !s_H5Acreate2 || !s_H5Awrite) return false;
  hid_t scalar_space = s_H5Screate_simple(0, nullptr, nullptr);
  if (scalar_space < 0) return false;
  hid_t attr = s_H5Acreate2(file, name, H5T_NATIVE_DOUBLE, scalar_space, 0, 0);
  bool ok = (attr >= 0) && (s_H5Awrite(attr, H5T_NATIVE_DOUBLE, &value) >= 0);
  if (attr >= 0) s_H5Aclose(attr);
  s_H5Sclose(scalar_space);
  return ok;
}

bool write_attr_int(hid_t file, const char* name, int value) {
  if (!available() || !s_H5Acreate2 || !s_H5Awrite) return false;
  hid_t scalar_space = s_H5Screate_simple(0, nullptr, nullptr);
  if (scalar_space < 0) return false;
  hid_t attr = s_H5Acreate2(file, name, H5T_NATIVE_INT, scalar_space, 0, 0);
  bool ok = (attr >= 0) && (s_H5Awrite(attr, H5T_NATIVE_INT, &value) >= 0);
  if (attr >= 0) s_H5Aclose(attr);
  s_H5Sclose(scalar_space);
  return ok;
}

bool write_attr_string(hid_t file, const char* name, const std::string& value) {
  // For simplicity, store as fixed-size array of chars
  if (!available() || !s_H5Acreate2 || !s_H5Awrite) return false;
  hsize_t dims[1] = { value.size() };
  hid_t space = s_H5Screate_simple(1, dims, nullptr);
  if (space < 0) return false;
  // Using native char type via H5T_NATIVE_INT8 would be more precise, but reuse INT
  hid_t attr = s_H5Acreate2(file, name, H5T_NATIVE_INT, space, 0, 0);
  bool ok = false;
  if (attr >= 0) {
    // Pad/copy into buffer of ints to keep simple; readers should treat as bytes
    std::vector<int> buf(value.begin(), value.end());
    ok = (s_H5Awrite(attr, H5T_NATIVE_INT, buf.data()) >= 0);
    s_H5Aclose(attr);
  }
  s_H5Sclose(space);
  return ok;
}

bool read_dataset_1d_double(hid_t file, const char* name, std::vector<double>& out) {
  if (!available() || !s_H5Dopen2 || !s_H5Dread || !s_H5Sget_simple_extent_dims) return false;
  hid_t dset = s_H5Dopen2(file, name, 0);
  if (dset < 0) return false;
  hid_t space = s_H5Dget_space ? s_H5Dget_space(dset) : -1;
  if (space < 0) { s_H5Dclose(dset); return false; }
  hsize_t dims[1] = {0};
  s_H5Sget_simple_extent_dims(space, dims, nullptr);
  out.resize(static_cast<size_t>(dims[0]));
  bool ok = (s_H5Dread(dset, H5T_NATIVE_DOUBLE, 0, 0, 0, out.data()) >= 0);
  s_H5Sclose(space);
  s_H5Dclose(dset);
  return ok;
}

bool write_dataset_1d_double(hid_t file, const char* name, const double* data, size_t n,
                             int /*compression_level*/, size_t /*chunk*/) {
  if (!available() || !s_H5Dcreate2 || !s_H5Screate_simple) return false;
  hsize_t dims[1] = { static_cast<hsize_t>(n) };
  hid_t space = s_H5Screate_simple(1, dims, nullptr);
  if (space < 0) return false;

  // Create without compression/chunking for simplicity and compatibility
  hid_t dset = s_H5Dcreate2(file, name, H5T_NATIVE_DOUBLE, space, 0, 0, 0);
  if (dset < 0) { s_H5Sclose(space); return false; }
  bool ok = (s_H5Dwrite(dset, H5T_NATIVE_DOUBLE, 0, 0, 0, data) >= 0);
  s_H5Dclose(dset);
  s_H5Sclose(space);
  return ok;
}

size_t dataset_length(hid_t file, const char* name) {
  if (!available() || !s_H5Dopen2 || !s_H5Sget_simple_extent_dims) return 0;
  hid_t dset = s_H5Dopen2(file, name, 0);
  if (dset < 0) return 0;
  hid_t space = s_H5Dget_space ? s_H5Dget_space(dset) : -1;
  if (space < 0) { s_H5Dclose(dset); return 0; }
  hsize_t dims[1] = {0};
  s_H5Sget_simple_extent_dims(space, dims, nullptr);
  s_H5Sclose(space);
  s_H5Dclose(dset);
  return static_cast<size_t>(dims[0]);
}

} // namespace h5rt
