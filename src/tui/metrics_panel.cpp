#include "tui/metrics_panel.hpp"
#include "core/utils.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>

#include <algorithm>

using namespace ftxui;

namespace neuralscope {

MetricsPanel::MetricsPanel() {}

void MetricsPanel::set_snapshot(const LayerSnapshot& snap) {
    current_  = snap;
    has_data_ = true;
}

void MetricsPanel::set_throughput(double tps) {
    throughput_ = tps;
}

void MetricsPanel::add_latency_sample(float latency_ms) {
    latency_history_.push_back(latency_ms);
    if (latency_history_.size() > MAX_HISTORY) {
        latency_history_.erase(latency_history_.begin());
    }
}

void MetricsPanel::add_stats_sample(float min_val, float mean_val, float max_val) {
    mean_history_.push_back(mean_val);
    max_history_.push_back(max_val);
    if (mean_history_.size() > MAX_HISTORY) {
        mean_history_.erase(mean_history_.begin());
    }
    if (max_history_.size() > MAX_HISTORY) {
        max_history_.erase(max_history_.begin());
    }
}

Component MetricsPanel::component() {
    return Renderer([this] { return render(); });
}

Element MetricsPanel::render() {
    if (!has_data_) {
        return vbox({
            text("  Waiting for data...") | dim | center,
        }) | flex | borderStyled(ROUNDED) | color(Color::GrayDark);
    }

    std::vector<Element> rows;

    // Tensor info
    rows.push_back(hbox({
        text(" Name         : ") | bold | color(Color::Cyan),
        text(current_.layer_name),
    }));

    rows.push_back(hbox({
        text(" Op           : ") | bold | color(Color::Cyan),
        text(current_.op_name),
    }));

    rows.push_back(hbox({
        text(" Tensor Shape : ") | bold | color(Color::Cyan),
        text(current_.tensor_shape),
    }));

    rows.push_back(hbox({
        text(" Dtype        : ") | bold | color(Color::Cyan),
        text(current_.dtype),
        text("  Elements: ") | dim,
        text(std::to_string(current_.n_elements)) | dim,
    }));

    rows.push_back(separator());

    // Sparsity blocks
    {
        float sp = current_.sparsity;
        
        // Build 20 blocks
        int total_blocks = 20;
        int filled = static_cast<int>(sp * total_blocks);
        
        std::string bar = "";
        for (int i = 0; i < total_blocks; ++i) {
            if (i < filled) {
                bar += (i > 15) ? "\xF0\x9F\x9F\xA5" /* Red */
                     : (i > 10) ? "\xF0\x9F\x9F\xA8" /* Yellow */
                     : "\xF0\x9F\x9F\xA9"; /* Green */
            } else {
                bar += "\xE2\xAC\x9C"; /* White/Empty */
            }
        }

        rows.push_back(hbox({
            text(" Sparsity Rate: ") | bold | color(Color::Cyan),
            text(bar + " " + format_float(sp * 100.0f, 1) + "%"),
        }));
    }

    // Latency
    {
        Color lat_col;
        int level = latency_color_level(current_.latency_ms);
        if (level == 0) lat_col = Color::Green;
        else if (level == 1) lat_col = Color::Yellow;
        else lat_col = Color::Red;

        rows.push_back(hbox({
            text(" Latency      : ") | bold | color(Color::Cyan),
            text(format_float(static_cast<float>(current_.latency_ms), 3) + " ms") |
                color(lat_col),
        }));
    }

    // Latency sparkline
    if (!latency_history_.empty()) {
        rows.push_back(hbox({
            text(" Latency Trend: ") | dim,
            text(sparkline(latency_history_, 30)) | color(Color::Cyan),
        }));
    }

    rows.push_back(separator());

    // Activation stats
    rows.push_back(hbox({
        text(" Act Min      : ") | bold | color(Color::Cyan),
        text(format_float(current_.act_min, 4)),
    }));
    rows.push_back(hbox({
        text(" Act Mean     : ") | bold | color(Color::Cyan),
        text(format_float(current_.act_mean, 4)),
    }));
    rows.push_back(hbox({
        text(" Act Max      : ") | bold | color(Color::Cyan),
        text(format_float(current_.act_max, 4)),
    }));
    rows.push_back(hbox({
        text(" Act Std      : ") | bold | color(Color::Cyan),
        text(format_float(current_.act_std, 4)),
    }));

    // Stats sparkline
    if (!mean_history_.empty()) {
        rows.push_back(hbox({
            text(" Mean Trend   : ") | dim,
            text(sparkline(mean_history_, 30)) | color(Color::Magenta1),
        }));
    }

    rows.push_back(separator());

    // Throughput
    rows.push_back(hbox({
        text(" Throughput   : ") | bold | color(Color::GreenLight),
        text(format_float(static_cast<float>(throughput_), 1) + " tok/s") |
            bold | color(Color::GreenLight),
    }));

    return vbox(std::move(rows)) | flex |
           borderStyled(ROUNDED) | color(Color::GrayDark);
}

} // namespace neuralscope
