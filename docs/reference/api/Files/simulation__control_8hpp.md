---
title: include/simulation/simulation_control.hpp

---

# include/simulation/simulation_control.hpp



## Functions

|                | Name           |
| -------------- | -------------- |
| bool | **[rollbackState](#function-rollbackstate)**(int n)<br>Roll back the simulation state by n iterations.  |


## Functions Documentation

### function rollbackState

```cpp
bool rollbackState(
    int n
)
```

Roll back the simulation state by n iterations. 

**Parameters**: 

  * **n** Number of iterations to roll back 


**Return**: true if rollback was successful, false otherwise 

This function reduces the size of simulation vectors by removing the last n iterations, effectively rolling back the simulation state to an earlier time point.




## Source code

```cpp
#ifndef SIMULATION_CONTROL_HPP
#define SIMULATION_CONTROL_HPP


bool rollbackState(int n);

#endif // SIMULATION_CONTROL_HPP
```


-------------------------------

Updated on 2025-10-03 at 23:06:53 +0200
