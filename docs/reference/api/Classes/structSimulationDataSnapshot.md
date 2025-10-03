---
title: SimulationDataSnapshot

---

# SimulationDataSnapshot





## Public Attributes

|                | Name           |
| -------------- | -------------- |
| std::vector< double > | **[QKv](#variable-qkv)**  |
| std::vector< double > | **[QRv](#variable-qrv)**  |
| std::vector< double > | **[dQKv](#variable-dqkv)**  |
| std::vector< double > | **[dQRv](#variable-dqrv)**  |
| std::vector< double > | **[t1grid](#variable-t1grid)**  |
| std::vector< double > | **[rvec](#variable-rvec)**  |
| std::vector< double > | **[drvec](#variable-drvec)**  |
| std::vector< double > | **[QKB1int](#variable-qkb1int)**  |
| std::vector< double > | **[QRB1int](#variable-qrb1int)**  |
| std::vector< double > | **[theta](#variable-theta)**  |
| double | **[energy](#variable-energy)**  |
| double | **[t_current](#variable-t-current)**  |
| int | **[current_len](#variable-current-len)**  |
| int | **[current_loop](#variable-current-loop)**  |
| SimulationConfig | **[config_snapshot](#variable-config-snapshot)**  |
| std::string | **[code_version](#variable-code-version)**  |
| std::string | **[git_hash](#variable-git-hash)**  |
| std::string | **[git_branch](#variable-git-branch)**  |
| std::string | **[git_tag](#variable-git-tag)**  |
| std::string | **[build_date](#variable-build-date)**  |
| std::string | **[build_time](#variable-build-time)**  |
| std::string | **[compiler_version](#variable-compiler-version)**  |
| std::string | **[cuda_version](#variable-cuda-version)**  |
| bool | **[git_dirty](#variable-git-dirty)**  |
| size_t | **[peak_memory_kb_snapshot](#variable-peak-memory-kb-snapshot)**  |
| size_t | **[peak_gpu_memory_mb_snapshot](#variable-peak-gpu-memory-mb-snapshot)**  |
| std::chrono::high_resolution_clock::time_point | **[program_start_time_snapshot](#variable-program-start-time-snapshot)**  |

## Public Attributes Documentation

### variable QKv

```cpp
std::vector< double > QKv;
```


### variable QRv

```cpp
std::vector< double > QRv;
```


### variable dQKv

```cpp
std::vector< double > dQKv;
```


### variable dQRv

```cpp
std::vector< double > dQRv;
```


### variable t1grid

```cpp
std::vector< double > t1grid;
```


### variable rvec

```cpp
std::vector< double > rvec;
```


### variable drvec

```cpp
std::vector< double > drvec;
```


### variable QKB1int

```cpp
std::vector< double > QKB1int;
```


### variable QRB1int

```cpp
std::vector< double > QRB1int;
```


### variable theta

```cpp
std::vector< double > theta;
```


### variable energy

```cpp
double energy;
```


### variable t_current

```cpp
double t_current;
```


### variable current_len

```cpp
int current_len;
```


### variable current_loop

```cpp
int current_loop;
```


### variable config_snapshot

```cpp
SimulationConfig config_snapshot;
```


### variable code_version

```cpp
std::string code_version;
```


### variable git_hash

```cpp
std::string git_hash;
```


### variable git_branch

```cpp
std::string git_branch;
```


### variable git_tag

```cpp
std::string git_tag;
```


### variable build_date

```cpp
std::string build_date;
```


### variable build_time

```cpp
std::string build_time;
```


### variable compiler_version

```cpp
std::string compiler_version;
```


### variable cuda_version

```cpp
std::string cuda_version;
```


### variable git_dirty

```cpp
bool git_dirty;
```


### variable peak_memory_kb_snapshot

```cpp
size_t peak_memory_kb_snapshot;
```


### variable peak_gpu_memory_mb_snapshot

```cpp
size_t peak_gpu_memory_mb_snapshot;
```


### variable program_start_time_snapshot

```cpp
std::chrono::high_resolution_clock::time_point program_start_time_snapshot;
```


-------------------------------

Updated on 2025-10-03 at 23:06:50 +0200