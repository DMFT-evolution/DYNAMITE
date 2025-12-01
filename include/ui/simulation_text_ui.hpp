// DYNAMITE: Text-based simulation UI helpers
#pragma once

#include <chrono>
#include "io/io_utils.hpp" // SaveTelemetry dependency
#include <cstddef>
#include <string>

namespace dmfe { namespace ui {

class SimulationTextUI {
public:
    SimulationTextUI();

    // Install SIGINT/SIGTERM (and SIGHUP if available) handlers to restore terminal state
    void install_signal_handlers() const;

    // Mark the start time for elapsed clock and ETA
    void start_clock();

    // Continuous multi-line status UI pinned to bottom. Throttled internally (~5 Hz).
    // Provide basic scalar state needed to render; method is a short identifier (e.g., "RK54").
    void update_status(double t, double tmax, int loop, double delta_t, const std::string& method);

    // One-line debug/verbose prints for TTY or non-TTY modes
    void print_debug_line(double t, int loop, double delta_t, double delta, const std::string& method,
                          double qk0, std::size_t current_t1len);
    void print_periodic_line(double t, int loop, double delta_t, double delta, const std::string& method,
                             double qk0, std::size_t current_t1len);

    // Final block printing and terminal cleanup. Will reset any scroll-region used.
    void print_final_results(bool aborted_by_signal,
                             double delta_t, double delta, int loop,
                             double t1_last, double rvec_last, double drvec_last,
                             double qkv_last, double qrv_last,
                             bool continuous_status);

    // Check if an interrupt signal was received
    bool interrupted() const;

private:
    // Helpers
    static int query_terminal_rows();
    static int query_terminal_cols();
    void set_scroll_region(int rows);
    int ensure_scroll_region();
    void reset_scroll_region();
    static std::string format_hms(double secs);
    static std::string make_progress_bar(double frac, int width);
    static std::string make_mem_bar(std::size_t used_kb, std::size_t total_kb, int width);

private:
    // Internal state for throttling and terminal region mgmt
    std::chrono::high_resolution_clock::time_point program_start_time_;
    std::chrono::high_resolution_clock::time_point last_status_print_;
    bool tui_scroll_region_set_ = false;
    int last_rows_ = -1;
};

}} // namespace dmfe::ui
