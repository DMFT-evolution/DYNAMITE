#include "version/version_compat.hpp"
#include "core/config.hpp"
#include <fstream>
#include <iostream>
#include "core/console.hpp"
#include <algorithm>
#include <sstream>
#include <vector>

// External declaration for global config variable
extern SimulationConfig config;

// Helper function to parse version string into vector of integers
std::vector<int> parseVersion(const std::string& version) {
    std::vector<int> parts;
    std::string v = version;
    if (v.substr(0, 1) == "v") v = v.substr(1);
    std::stringstream ss(v);
    std::string part;
    while (std::getline(ss, part, '.')) {
        try {
            parts.push_back(std::stoi(part));
        } catch (...) {
            parts.push_back(0);
        }
    }
    return parts;
}

// Check if two versions are compatible (first two positions match)
bool areVersionsCompatible(const std::string& v1, const std::string& v2) {
    auto p1 = parseVersion(v1);
    auto p2 = parseVersion(v2);
    if (p1.size() < 2 || p2.size() < 2) return false;
    return p1[0] == p2[0] && p1[1] == p2[1];
}

VersionAnalysis analyzeVersionCompatibility(const std::string& paramFilename) {
    VersionAnalysis analysis;
    analysis.current_version = g_version_info.code_version;
    std::ifstream paramFile(paramFilename);
    std::string line;
    std::string file_git_hash, file_git_branch;
    bool file_git_dirty = false;
    while (std::getline(paramFile, line)) {
        if (line.find("code_version = ") == 0) {
            analysis.file_version = line.substr(15);
        } else if (line.find("git_hash = ") == 0) {
            file_git_hash = line.substr(11);
        } else if (line.find("git_branch = ") == 0) {
            file_git_branch = line.substr(13);
        } else if (line.find("git_dirty = ") == 0) {
            file_git_dirty = (line.substr(12) == "true");
        }
    }
    if (analysis.file_version == "unknown") {
        analysis.level = VersionCompatibility::WARNING;
        analysis.warnings.push_back("File version information not found");
        return analysis;
    }
    if (analysis.file_version == analysis.current_version) {
        analysis.level = VersionCompatibility::IDENTICAL;
        return analysis;
    }
    // Check if versions are compatible (first two positions match)
    if (areVersionsCompatible(analysis.file_version, analysis.current_version)) {
        // Additional checks for warnings
        if (file_git_hash == g_version_info.git_hash) {
            if (file_git_dirty != g_version_info.git_dirty) {
                analysis.warnings.push_back("Same commit, different dirty state");
            }
        } else if (file_git_branch == g_version_info.git_branch && file_git_branch != "unknown") {
            analysis.warnings.push_back("Same branch, different commits");
            analysis.warnings.push_back("File: " + file_git_hash.substr(0, 7) + ", Current: " + g_version_info.git_hash.substr(0, 7));
        }
        analysis.level = VersionCompatibility::COMPATIBLE;
        return analysis;
    } else {
        analysis.level = VersionCompatibility::INCOMPATIBLE;
        analysis.errors.push_back("Version incompatibility detected");
        analysis.errors.push_back("File version: " + analysis.file_version + ", Current version: " + analysis.current_version);
        return analysis;
    }
}

// Forward declarations of helpers expected elsewhere
VersionAnalysis checkVersionCompatibility(const std::string& paramFilename);

bool checkVersionCompatibilityInteractive(const std::string& paramFilename) {
    VersionAnalysis analysis = analyzeVersionCompatibility(paramFilename);
    switch (analysis.level) {
        case VersionCompatibility::IDENTICAL:
        case VersionCompatibility::COMPATIBLE:
            return true;
        case VersionCompatibility::WARNING:
            std::cerr << dmfe::console::WARN() << "Version warning: potential differences detected. Proceeding." << std::endl;
            for (auto &w : analysis.warnings) std::cerr << dmfe::console::WARN() << "  - " << w << std::endl;
            return true;
        case VersionCompatibility::INCOMPATIBLE:
            if (config.allow_incompatible_versions) {
                std::cerr << dmfe::console::WARN() << "Version incompatibility detected, but proceeding due to --allow-incompatible-versions flag." << std::endl;
                for (auto &e : analysis.errors) std::cerr << dmfe::console::ERR() << "  - " << e << std::endl;
                return true;
            } else {
                std::cerr << dmfe::console::ERR() << "Version incompatibility detected. Aborting load." << std::endl;
                for (auto &e : analysis.errors) std::cerr << dmfe::console::ERR() << "  - " << e << std::endl;
                return false;
            }
    }
    return false;
}

// Basic version compatibility check (simple string comparison)
bool checkVersionCompatibilityBasic(const std::string& paramFilename) {
    std::ifstream paramFile(paramFilename);
    std::string line;
    std::string file_version = "unknown";
    
    while (std::getline(paramFile, line)) {
        if (line.find("code_version = ") == 0) {
            file_version = line.substr(15);
            break;
        }
    }
    
    if (file_version != g_version_info.code_version) {
    std::cout << dmfe::console::WARN() << "Parameter file created with version " << file_version 
          << ", current version is " << g_version_info.code_version << std::endl;
        return false;
    }
    return true;
}
