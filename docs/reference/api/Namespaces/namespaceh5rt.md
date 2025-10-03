---
title: h5rt

---

# h5rt



## Types

|                | Name           |
| -------------- | -------------- |
| using long long | **[hid_t](#using-hid-t)**  |
| using int | **[herr_t](#using-herr-t)**  |
| using unsigned long long | **[hsize_t](#using-hsize-t)**  |

## Functions

|                | Name           |
| -------------- | -------------- |
| bool | **[try_load](#function-try-load)**() |
| bool | **[available](#function-available)**() |
| void | **[unload](#function-unload)**() |
| hid_t | **[open_file_readonly](#function-open-file-readonly)**(const char * path) |
| hid_t | **[create_file_trunc](#function-create-file-trunc)**(const char * path) |
| void | **[close_file](#function-close-file)**(hid_t file) |
| bool | **[read_attr_double](#function-read-attr-double)**(hid_t file, const char * name, double & out) |
| bool | **[read_attr_int](#function-read-attr-int)**(hid_t file, const char * name, int & out) |
| bool | **[write_attr_double](#function-write-attr-double)**(hid_t file, const char * name, double value) |
| bool | **[write_attr_int](#function-write-attr-int)**(hid_t file, const char * name, int value) |
| bool | **[write_attr_string](#function-write-attr-string)**(hid_t file, const char * name, const std::string & value) |
| bool | **[read_dataset_1d_double](#function-read-dataset-1d-double)**(hid_t file, const char * name, std::vector< double > & out) |
| bool | **[write_dataset_1d_double](#function-write-dataset-1d-double)**(hid_t file, const char * name, const double * data, size_t n, int compression_level =6, size_t chunk =4096) |
| size_t | **[dataset_length](#function-dataset-length)**(hid_t file, const char * name) |

## Types Documentation

### using hid_t

```cpp
using h5rt::hid_t = typedef long long;
```


### using herr_t

```cpp
using h5rt::herr_t = typedef int;
```


### using hsize_t

```cpp
using h5rt::hsize_t = typedef unsigned long long;
```



## Functions Documentation

### function try_load

```cpp
bool try_load()
```


### function available

```cpp
bool available()
```


### function unload

```cpp
void unload()
```


### function open_file_readonly

```cpp
hid_t open_file_readonly(
    const char * path
)
```


### function create_file_trunc

```cpp
hid_t create_file_trunc(
    const char * path
)
```


### function close_file

```cpp
void close_file(
    hid_t file
)
```


### function read_attr_double

```cpp
bool read_attr_double(
    hid_t file,
    const char * name,
    double & out
)
```


### function read_attr_int

```cpp
bool read_attr_int(
    hid_t file,
    const char * name,
    int & out
)
```


### function write_attr_double

```cpp
bool write_attr_double(
    hid_t file,
    const char * name,
    double value
)
```


### function write_attr_int

```cpp
bool write_attr_int(
    hid_t file,
    const char * name,
    int value
)
```


### function write_attr_string

```cpp
bool write_attr_string(
    hid_t file,
    const char * name,
    const std::string & value
)
```


### function read_dataset_1d_double

```cpp
bool read_dataset_1d_double(
    hid_t file,
    const char * name,
    std::vector< double > & out
)
```


### function write_dataset_1d_double

```cpp
bool write_dataset_1d_double(
    hid_t file,
    const char * name,
    const double * data,
    size_t n,
    int compression_level =6,
    size_t chunk =4096
)
```


### function dataset_length

```cpp
size_t dataset_length(
    hid_t file,
    const char * name
)
```






-------------------------------

Updated on 2025-10-03 at 23:06:50 +0200