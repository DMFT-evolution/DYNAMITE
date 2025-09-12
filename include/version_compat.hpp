#pragma once
#include "version_info.hpp"
#include <string>

VersionAnalysis analyzeVersionCompatibility(const std::string& paramFilename);
bool checkVersionCompatibilityInteractive(const std::string& paramFilename);
VersionAnalysis checkVersionCompatibility(const std::string& paramFilename);
bool checkVersionCompatibilityBasic(const std::string& paramFilename);
