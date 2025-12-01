#include "ui/simulation_text_ui.hpp"
#include "core/console.hpp"
#include "core/config.hpp"
#include "simulation/simulation_data.hpp"
#include "io/io_utils.hpp"
#include <csignal>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cmath>

// External globals used for memory and telemetry (declared elsewhere)
extern SimulationConfig config;
extern SimulationData* sim;
extern size_t peak_memory_kb;
#if DMFE_WITH_CUDA
extern size_t peak_gpu_memory_mb;
extern size_t getAvailableGPUMemory();
extern size_t getGPUMemoryUsage();
#endif
extern size_t getCurrentMemoryUsage();
extern size_t getTotalSystemMemoryKB();
// Telemetry helpers are declared in io/io_utils.hpp

namespace {
    // Async-signal-safe interrupted flag
    volatile sig_atomic_t g_interrupted_ui = 0;
    void dmfe_sig_handler_ui(int /*sig*/) {
        g_interrupted_ui = 1;
        const char seq1[] = "\033[r";
        const char seq2[] = "\033[?25h";
        const char seq3[] = "\033[999;1H";
        const char seq4[] = "\033[K";
        const char seq5[] = "\n";
        (void)!write(STDOUT_FILENO, seq1, sizeof(seq1)-1);
        (void)!write(STDOUT_FILENO, seq2, sizeof(seq2)-1);
        (void)!write(STDOUT_FILENO, seq3, sizeof(seq3)-1);
        (void)!write(STDOUT_FILENO, seq4, sizeof(seq4)-1);
        (void)!write(STDOUT_FILENO, seq5, sizeof(seq5)-1);
    }
}

namespace dmfe { namespace ui {

SimulationTextUI::SimulationTextUI() {
    last_status_print_ = std::chrono::high_resolution_clock::now();
}

void SimulationTextUI::install_signal_handlers() const {
    std::signal(SIGINT, dmfe_sig_handler_ui);
    std::signal(SIGTERM, dmfe_sig_handler_ui);
#ifdef SIGHUP
    std::signal(SIGHUP, dmfe_sig_handler_ui);
#endif
}

void SimulationTextUI::start_clock() {
    program_start_time_ = std::chrono::high_resolution_clock::now();
}

bool SimulationTextUI::interrupted() const { return g_interrupted_ui != 0; }

int SimulationTextUI::query_terminal_rows() {
    struct winsize ws{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0) return ws.ws_row;
    if (const char* env = std::getenv("LINES")) {
        int v = std::atoi(env); if (v > 0) return v;
    }
    return 24;
}

int SimulationTextUI::query_terminal_cols() {
    struct winsize ws{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) return ws.ws_col;
    if (const char* env = std::getenv("COLUMNS")) {
        int v = std::atoi(env); if (v > 0) return v;
    }
    return 80;
}

void SimulationTextUI::set_scroll_region(int rows) {
    if (rows >= 6) {
        std::cout << "\033[r"; // reset
        std::cout << "\033[1;" << (rows - 4) << "r";
        tui_scroll_region_set_ = true;
        last_rows_ = rows;
    } else {
        std::cout << "\033[r";
        tui_scroll_region_set_ = false;
        last_rows_ = -1;
    }
}

int SimulationTextUI::ensure_scroll_region() {
    int rows = query_terminal_rows();
    if (!tui_scroll_region_set_ || rows != last_rows_) set_scroll_region(rows);
    return rows;
}

void SimulationTextUI::reset_scroll_region() {
    if (tui_scroll_region_set_) {
        std::cout << "\033[r" << std::flush;
        tui_scroll_region_set_ = false;
        last_rows_ = -1;
    }
}

std::string SimulationTextUI::format_hms(double secs) {
    if (!std::isfinite(secs) || secs < 0) secs = 0;
    long s = static_cast<long>(secs + 0.5);
    long h = s / 3600; s %= 3600;
    long m = s / 60;   s %= 60;
    std::ostringstream os; os.fill('0');
    os << std::setw(2) << h << ":" << std::setw(2) << m << ":" << std::setw(2) << s;
    return os.str();
}

std::string SimulationTextUI::make_progress_bar(double frac, int width) {
    frac = std::max(0.0, std::min(1.0, frac));
    width = std::max(10, width);
    int filled = static_cast<int>(std::round(frac * width));
    filled = std::clamp(filled, 0, width);
    std::string bar; bar.reserve(static_cast<size_t>(width)+2);
    bar.push_back('[');
    bar.append(static_cast<size_t>(filled), '=');
    bar.append(static_cast<size_t>(width - filled), ' ');
    bar.push_back(']');
    return bar;
}

std::string SimulationTextUI::make_mem_bar(std::size_t used_kb, std::size_t total_kb, int width) {
    if (total_kb == 0) return {};
    width = std::max(10, width);
    double frac = std::min(1.0, (double)used_kb / (double)total_kb);
    int filled = static_cast<int>(std::floor(frac * width));
    if (frac > 0.0 && filled == 0) filled = 1;
    filled = std::clamp(filled, 0, width);
    std::string bar; bar.reserve(static_cast<size_t>(width)+2);
    bar.push_back('[');
    bar.append(static_cast<size_t>(filled), '=');
    bar.append(static_cast<size_t>(width - filled), ' ');
    bar.push_back(']');
    const bool enable_color = dmfe::console::color_out();
    const char* color = dmfe::console::C_GREEN;
    if (frac >= 0.75) color = dmfe::console::C_RED; else if (frac >= 0.50) color = dmfe::console::C_YELLOW;
    if (enable_color) return std::string(color) + bar + dmfe::console::C_RESET;
    return bar;
}

void SimulationTextUI::update_status(double t, double tmax, int loop, double delta_t, const std::string& method) {
    const bool continuous_status = (!config.debug) && dmfe::console::stdout_is_tty();
    if (!continuous_status) return; // Only multi-line TUI in continuous mode
    auto now = std::chrono::high_resolution_clock::now();
    bool force_refresh = consumeSaveTelemetryDirty();
    bool reset_anchor = consumeStatusAnchorInvalidated();
    if (reset_anchor) dmfe::console::end_status_line_if_needed(true);
    if (!force_refresh && std::chrono::duration_cast<std::chrono::milliseconds>(now - last_status_print_).count() < 200) return;
    last_status_print_ = now;

    double elapsed_s = std::chrono::duration<double>(now - program_start_time_).count();
    double frac = (tmax > 0.0) ? std::clamp(t / tmax, 0.0, 1.0) : 0.0;
    double eta_s = (frac > 1e-6) ? elapsed_s * (1.0 - frac) / frac : std::numeric_limits<double>::infinity();
    std::string bar = make_progress_bar(frac, 28);

    // Memory info
    size_t cur_kb = 0; try { cur_kb = getCurrentMemoryUsage(); } catch (...) { cur_kb = 0; }
    size_t total_kb = getTotalSystemMemoryKB();
    std::ostringstream mem;
    if (cur_kb && total_kb) {
        double frac_ram = std::min(1.0, (double)cur_kb / (double)total_kb);
        const char* ram_color = dmfe::console::C_GREEN;
        if (frac_ram >= 0.75) ram_color = dmfe::console::C_RED; else if (frac_ram >= 0.50) ram_color = dmfe::console::C_YELLOW;
        if (dmfe::console::color_out()) mem << ram_color;
        mem << "RAM " << (cur_kb/1024) << "/" << (total_kb/1024) << " MB " << make_mem_bar(cur_kb, total_kb, 20);
        if (dmfe::console::color_out()) mem << dmfe::console::C_RESET;
    } else if (cur_kb) {
        mem << "RAM " << (cur_kb/1024) << " MB";
    } else {
        mem << "RAM peak " << (peak_memory_kb/1024) << " MB";
    }
#if DMFE_WITH_CUDA
    if (config.gpu) {
        size_t total_gpu_mb = getAvailableGPUMemory();
        size_t used_gpu_mb = getGPUMemoryUsage();
        if (total_gpu_mb && used_gpu_mb) {
            double frac_gpu = std::min(1.0, (double)used_gpu_mb / (double)total_gpu_mb);
            const char* gpu_color = dmfe::console::C_GREEN;
            if (frac_gpu >= 0.75) gpu_color = dmfe::console::C_RED; else if (frac_gpu >= 0.50) gpu_color = dmfe::console::C_YELLOW;
            mem << " | "; if (dmfe::console::color_out()) mem << gpu_color;
            mem << "GPU " << used_gpu_mb << "/" << total_gpu_mb << " MB "
                << make_mem_bar(used_gpu_mb*1024ULL, total_gpu_mb*1024ULL, 20);
            if (dmfe::console::color_out()) mem << dmfe::console::C_RESET;
            mem << " (peak " << peak_gpu_memory_mb << "MB)";
        } else {
            mem << " | GPU used " << used_gpu_mb << " MB (peak " << peak_gpu_memory_mb << "MB)";
        }
    }
#endif

    SaveTelemetry st = getSaveTelemetry();
    std::ostringstream save_line;
    if (!config.save_output) {
        save_line << "(save disabled)";
    } else if (st.in_progress) {
        double since = std::chrono::duration<double>(now - st.last_start_time).count();
        std::string pbar = make_progress_bar(st.progress, 20);
        save_line.setf(std::ios::fixed);
        save_line << "saving " << (st.stage.empty()?"...":st.stage) << " " << pbar
                  << " | t_saved " << std::setprecision(6) << st.last_t_exported
                  << " | elapsed " << format_hms(since);
        if (config.debug && !st.target_file.empty()) save_line << " -> " << st.target_file;
    } else if (!st.last_completed_file.empty()) {
        double ago = (st.last_end_time.time_since_epoch().count() > 0)
            ? std::chrono::duration<double>(now - st.last_end_time).count() : 0.0;
        std::string dir = st.last_completed_file;
        auto pos = dir.find_last_of('/'); if (pos != std::string::npos) dir = dir.substr(0, pos);
        save_line << "last save: " << dir << " (" << (ago>0?format_hms(ago):"now") << ")";
        if (config.debug) save_line << " | file " << st.last_completed_file;
    } else {
        save_line << "save: none yet";
    }

    std::ostringstream line1, line2, line3, line4;
    line1.setf(std::ios::fixed);
    line1 << " time " << format_hms(elapsed_s) << " | sim " << std::setprecision(6) << t
          << "/" << tmax << " (" << std::setprecision(1) << (frac * 100.0) << "%) " << bar;
    line2.setf(std::ios::fixed);
    line2 << " dt " << std::setprecision(6) << delta_t << " | meth " << method << " | loop " << loop;
    if (std::isfinite(eta_s)) line2 << " | ETA " << format_hms(eta_s);
    line3 << mem.str();
    line4 << save_line.str();

    int rows = ensure_scroll_region();
    int cols = query_terminal_cols();
    if (rows > 6) {
        int base = rows - 3;
        std::cout << "\0337\033[s"; // save cursor
        std::cout << "\033[" << base << ";1H" << dmfe::console::STAT() << line1.str() << "\033[K";
        std::cout << "\033[" << (base+1) << ";1H" << dmfe::console::STAT() << line2.str() << "\033[K";
        std::cout << "\033[" << (base+2) << ";1H" << dmfe::console::STAT() << line3.str() << "\033[K";
        {
            std::string save_text = line4.str();
            int max_text_width = std::max(0, cols - 9);
            if (max_text_width > 0 && static_cast<int>(save_text.size()) > max_text_width)
                save_text.erase(static_cast<size_t>(max_text_width));
            std::cout << "\033[" << (base+3) << ";1H" << dmfe::console::STAT() << save_text << "\033[K";
        }
        std::cout << "\0338\033[u"; // restore cursor
        std::cout << "\033[" << (rows - 4) << ";1H" << std::flush;
    } else {
        std::cout << dmfe::console::STAT() << line1.str() << " | " << line2.str() << "\n" << std::flush;
    }
}

void SimulationTextUI::print_debug_line(double t, int loop, double delta_t, double delta, const std::string& method,
                                        double qk0, std::size_t current_t1len) {
    std::ostringstream oss; oss.setf(std::ios::fixed); oss << std::setprecision(14)
        << " loop: " << loop
        << " time: " << t
        << " time step: " << delta_t
        << " delta: " << delta
        << " method: " << method
        << " QK: " << qk0
        << " length of t1grid: " << current_t1len;
    std::cout << dmfe::console::STAT() << oss.str() << std::endl;
}

void SimulationTextUI::print_periodic_line(double t, int loop, double delta_t, double delta, const std::string& method,
                                           double qk0, std::size_t current_t1len) {
    auto now = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_status_print_).count() < 1) return;
    last_status_print_ = now;
    print_debug_line(t, loop, delta_t, delta, method, qk0, current_t1len);
}

void SimulationTextUI::print_final_results(bool aborted_by_signal,
                                           double delta_t, double delta, int loop,
                                           double t1_last, double rvec_last, double drvec_last,
                                           double qkv_last, double qrv_last,
                                           bool continuous_status) {
    dmfe::console::end_status_line_if_needed(continuous_status);
    reset_scroll_region();
    if (dmfe::console::stdout_is_tty()) std::cout << "\033[999;1H";
    if (!aborted_by_signal) {
        std::cout << dmfe::console::RES() << "final delta_t: " << delta_t << std::endl;
        std::cout << dmfe::console::RES() << "final delta:   " << delta << std::endl;
        std::cout << dmfe::console::RES() << "final loop:    " << loop << std::endl;
        std::cout << dmfe::console::RES() << "final t1grid:  " << t1_last << std::endl;
        std::cout << dmfe::console::RES() << "final rvec:    " << rvec_last << std::endl;
        std::cout << dmfe::console::RES() << "final drvec:   " << drvec_last << std::endl;
        std::cout << dmfe::console::RES() << "final QKv:     " << qkv_last << std::endl;
        std::cout << dmfe::console::RES() << "final QRv:     " << qrv_last << std::endl;
        std::cout << dmfe::console::DONE() << "Simulation finished." << std::endl;
    } else {
        std::cout << dmfe::console::DONE() << "Simulation aborted by user (SIGINT)." << std::endl;
    }
}

}} // namespace dmfe::ui
