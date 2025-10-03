#ifndef SIMULATION_CONTROL_HPP
#define SIMULATION_CONTROL_HPP


/**
 * @brief Roll back the simulation state by n iterations
 * 
 * This function reduces the size of simulation vectors by removing the last n iterations,
 * effectively rolling back the simulation state to an earlier time point.
 * 
 * @param n Number of iterations to roll back
 * @return true if rollback was successful, false otherwise
 */
bool rollbackState(int n);

#endif // SIMULATION_CONTROL_HPP
