#pragma once

#include "core/types.hpp"
#include "core/ring_buffer.hpp"
#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>
#include <vector>

namespace neuralscope {

/// Panel 4: Runtime metrics inspector with gauges and sparklines.
class MetricsPanel {
public:
    MetricsPanel();

    /// Update with the currently selected snapshot.
    void set_snapshot(const LayerSnapshot& snap);

    /// Set tokens per second.
    void set_throughput(double tps);

    /// Add a latency value to the sparkline history.
    void add_latency_sample(float latency_ms);

    /// Add activation stats to sparkline history.
    void add_stats_sample(float min_val, float mean_val, float max_val);

    /// Get FTXUI component.
    ftxui::Component component();

    /// Get FTXUI element for rendering.
    ftxui::Element render();

private:
    LayerSnapshot current_;
    double        throughput_ = 0.0;
    bool          has_data_   = false;

    // Sparkline histories
    std::vector<float> latency_history_;
    std::vector<float> mean_history_;
    std::vector<float> max_history_;
    static constexpr size_t MAX_HISTORY = 50;
};

} // namespace neuralscope
