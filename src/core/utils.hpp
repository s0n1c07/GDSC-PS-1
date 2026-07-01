#pragma once

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>
#include <fstream>

namespace neuralscope {

/// Get current timestamp as HH:MM:SS.mmm
inline std::string now_timestamp() {
    auto now   = std::chrono::system_clock::now();
    auto time  = std::chrono::system_clock::to_time_t(now);
    auto ms    = std::chrono::duration_cast<std::chrono::milliseconds>(
                     now.time_since_epoch()) % 1000;

    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &time);
#else
    localtime_r(&time, &tm_buf);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

/// Get current timestamp as ISO format for export
inline std::string now_iso_timestamp() {
    auto now   = std::chrono::system_clock::now();
    auto time  = std::chrono::system_clock::to_time_t(now);

    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &time);
#else
    localtime_r(&time, &tm_buf);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

/// Format a float with fixed precision
inline std::string format_float(float value, int precision = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

/// Format latency with color hint: 0=green, 1=yellow, 2=red
inline int latency_color_level(double ms) {
    if (ms < 1.0) return 0;   // green
    if (ms < 5.0) return 1;   // yellow
    return 2;                  // red
}

/// Build a sparsity gauge bar using Unicode blocks
/// Returns something like: "████████░░░░ 67.3%"
inline std::string sparsity_gauge(float sparsity, int width = 12) {
    int filled = static_cast<int>(std::round(sparsity * width));
    std::string bar;
    for (int i = 0; i < width; ++i) {
        if (i < filled) {
            bar += "\xe2\x96\x88"; // █
        } else {
            bar += "\xe2\x96\x91"; // ░
        }
    }
    bar += " " + format_float(sparsity * 100.0f, 1) + "%";
    return bar;
}

/// Build a simple sparkline from a vector of values
/// Uses Unicode block characters: ▁▂▃▄▅▆▇█
inline std::string sparkline(const std::vector<float>& values, size_t max_width = 30) {
    if (values.empty()) return "";

    const char* blocks[] = {
        "\xe2\x96\x81", "\xe2\x96\x82", "\xe2\x96\x83", "\xe2\x96\x84",
        "\xe2\x96\x85", "\xe2\x96\x86", "\xe2\x96\x87", "\xe2\x96\x88"
    };

    // Use last max_width values
    size_t start = values.size() > max_width ? values.size() - max_width : 0;
    float min_val = values[start], max_val = values[start];
    for (size_t i = start; i < values.size(); ++i) {
        if (values[i] < min_val) min_val = values[i];
        if (values[i] > max_val) max_val = values[i];
    }

    float range = max_val - min_val;
    if (range < 1e-9f) range = 1.0f;

    std::string result;
    for (size_t i = start; i < values.size(); ++i) {
        int idx = static_cast<int>((values[i] - min_val) / range * 7.0f);
        idx = std::max(0, std::min(7, idx));
        result += blocks[idx];
    }
    return result;
}

/// Truncate string to max_len, adding "..." if needed
inline std::string truncate(const std::string& str, size_t max_len) {
    if (str.size() <= max_len) return str;
    if (max_len <= 3) return str.substr(0, max_len);
    return str.substr(0, max_len - 3) + "...";
}

/// Format tensor shape from ggml ne[] array
inline std::string format_shape(const int64_t ne[4]) {
    std::string result = "[";
    bool first = true;
    for (int i = 0; i < 4; ++i) {
        if (ne[i] == 1 && i > 0) continue; // Skip trailing 1s
        if (!first) result += ", ";
        result += std::to_string(ne[i]);
        first = false;
    }
    result += "]";
    return result;
}


inline void debug_log(const std::string& msg) {
    std::ofstream log("debug_run.log", std::ios::app);
    log << msg << std::endl;
}
} // namespace neuralscope
