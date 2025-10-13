#pragma once

#include <string>
#include <cstdlib>
#include <unistd.h>
#include <iostream>

namespace dmfe { namespace console {

inline bool& initialized() { static bool v = false; return v; }
inline bool& color_out() { static bool v = false; return v; }
inline bool& color_err() { static bool v = false; return v; }
inline bool& stdout_is_tty() { static bool v = false; return v; }

inline bool env_set_nonzero(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
}

inline void init() {
    if (initialized()) return;
    const bool force_color = env_set_nonzero("FORCE_COLOR");
    const bool no_color = std::getenv("NO_COLOR") != nullptr;
    stdout_is_tty() = ::isatty(STDOUT_FILENO) == 1;
    const bool stderr_is_tty = ::isatty(STDERR_FILENO) == 1;
    color_out() = !no_color && (force_color || stdout_is_tty());
    color_err() = !no_color && (force_color || stderr_is_tty);
    initialized() = true;
}

// ANSI color codes
constexpr const char* C_RESET  = "\033[0m";
constexpr const char* C_BOLD   = "\033[1m";
constexpr const char* C_RED    = "\033[31m";
constexpr const char* C_GREEN  = "\033[32m";
constexpr const char* C_YELLOW = "\033[33m"; // orange-ish
constexpr const char* C_BLUE   = "\033[34m";
constexpr const char* C_MAG    = "\033[35m";
constexpr const char* C_CYAN   = "\033[36m";
constexpr const char* C_GRAY   = "\033[38;2;160;160;160m";

inline const char* use(const char* code, bool enabled) { return enabled ? code : ""; }

inline std::string INFO() { init(); return std::string(use(C_GRAY, color_out())) + "[INFO] " + use(C_RESET, color_out()); }
inline std::string WARN() { init(); return std::string(use(C_YELLOW, color_out())) + "[WARN] " + use(C_RESET, color_out()); }
inline std::string ERR()  { init(); return std::string(use(C_RED, color_err())) + "[ERROR] " + use(C_RESET, color_err()); }
inline std::string SAVE() { init(); return std::string(use(C_BLUE, color_out())) + "[SAVE] " + use(C_RESET, color_out()); }
inline std::string STAT() { init(); return std::string(use(C_CYAN, color_out())) + "[STATUS] " + use(C_RESET, color_out()); }
inline std::string BENCH(){ init(); return std::string(use(C_MAG, color_out())) + "[BENCH] " + use(C_RESET, color_out()); }
inline std::string DONE() { init(); return std::string(use(C_GREEN, color_out())) + "[DONE] " + use(C_RESET, color_out()); }
inline std::string RES()  { init(); return std::string(use(C_BOLD, color_out())) + "[RESULT] " + use(C_RESET, color_out()); }

// Helper to end a status line before printing a regular line
inline void end_status_line_if_needed(bool continuous_status_active) {
    if (continuous_status_active) {
        std::cout << std::endl;
    }
}

}} // namespace dmfe::console
