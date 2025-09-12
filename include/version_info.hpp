#pragma once
#include <string>
#include <vector>

struct VersionInfo {
    std::string git_hash;
    std::string git_branch;
    std::string git_tag;
    bool git_dirty;
    std::string build_date;
    std::string build_time;
    std::string compiler_version;
    std::string cuda_version;
    std::string code_version;
    VersionInfo();
    std::string toString() const;
};

enum class VersionCompatibility {
    IDENTICAL,
    COMPATIBLE,
    WARNING,
    INCOMPATIBLE
};

struct VersionAnalysis {
    VersionCompatibility level;
    std::string file_version;
    std::string current_version;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
};

extern VersionInfo g_version_info;
