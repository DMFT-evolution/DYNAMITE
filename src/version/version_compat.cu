#include "version_compat.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

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
    if (file_git_hash == g_version_info.git_hash) {
        if (file_git_dirty != g_version_info.git_dirty) {
            analysis.level = VersionCompatibility::COMPATIBLE;
            analysis.warnings.push_back("Same commit, different dirty state");
        } else {
            analysis.level = VersionCompatibility::COMPATIBLE;
        }
        return analysis;
    }
    if (file_git_branch == g_version_info.git_branch && file_git_branch != "unknown") {
        analysis.level = VersionCompatibility::WARNING;
        analysis.warnings.push_back("Same branch, different commits");
        analysis.warnings.push_back("File: " + file_git_hash.substr(0, 7) + ", Current: " + g_version_info.git_hash.substr(0, 7));
        return analysis;
    }
    analysis.level = VersionCompatibility::INCOMPATIBLE;
    analysis.errors.push_back("Significant version difference detected");
    analysis.errors.push_back("File branch: " + file_git_branch + ", Current branch: " + g_version_info.git_branch);
    return analysis;
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
            std::cerr << "Version warning: potential differences detected. Proceeding." << std::endl;
            for (auto &w : analysis.warnings) std::cerr << "  - " << w << std::endl;
            return true;
        case VersionCompatibility::INCOMPATIBLE:
            std::cerr << "Version incompatibility detected. Aborting load." << std::endl;
            for (auto &e : analysis.errors) std::cerr << "  - " << e << std::endl;
            return false;
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
        std::cout << "Warning: Parameter file created with version " << file_version 
                  << ", current version is " << g_version_info.code_version << std::endl;
        return false;
    }
    return true;
}
