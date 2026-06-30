#ifndef TIME_STEPS_HPP
#define TIME_STEPS_HPP

#include "EOMs/time_steps_cpu.hpp"

// Public backend-independent interface

double rstep();
double drstep();

#endif // TIME_STEPS_HPP