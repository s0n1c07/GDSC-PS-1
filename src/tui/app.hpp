#pragma once

#include "core/types.hpp"
#include "core/ring_buffer.hpp"
#include "engine/inference.hpp"
#include "tui/startup_menu.hpp"
#include "tui/topology_tree.hpp"
#include "tui/packet_stream.hpp"
#include "tui/attention_heatmap.hpp"
#include "tui/metrics_panel.hpp"
#include "tui/anomaly_ledger.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace neuralscope {

/// Main TUI application that composes all panels into the dashboard layout.
class App {
public:
    App();
    ~App();

    /// Run the application (blocks until quit).
    void run();

    /// Pre-configure and skip the startup menu (used when CLI flags are given).
    void launch_with_config(const AppConfig& config,
                             const std::string& initial_prompt = "");

private:
    /// Build the startup menu screen.
    ftxui::Component build_startup_screen();

    /// Build the main inspector dashboard.
    ftxui::Component build_dashboard();

    /// Handle the launch action from startup menu.
    void on_launch(const AppConfig& config);

    /// Update panels with latest data from ring buffer / replay state.
    void refresh_panels();

    // ─── Core state ────────────────────────────────────────────
    ftxui::ScreenInteractive    screen_;
    AppConfig                   config_;
    bool                        in_dashboard_ = false;

    // ─── Data buffers ──────────────────────────────────────────
    RingBuffer<LayerSnapshot>   snapshot_buffer_;
    RingBuffer<AnomalyEntry>    anomaly_buffer_;

    // ─── Engine ────────────────────────────────────────────────
    std::unique_ptr<InferenceEngine> engine_;

    // ─── TUI panels ────────────────────────────────────────────
    StartupMenu                 startup_menu_;
    TopologyTree                topology_tree_;
    PacketStream                packet_stream_;
    AttentionHeatmap            attention_heatmap_;
    MetricsPanel                metrics_panel_;
    AnomalyLedger               anomaly_ledger_;

    // ─── Dashboard state ───────────────────────────────────────
    int                         screen_index_ = 0; // 0 = startup, 1 = dashboard
    int                         active_panel_ = 0;
    int                         focused_panel_ = 0;
    std::string                 prompt_text_;
    std::string                 status_text_ = "Ready";
    static constexpr int        NUM_PANELS = 5;

    // ─── Replay state ──────────────────────────────────────────
    std::vector<LayerSnapshot>  replay_snapshots_;   // Loaded from capture file
    size_t                      replay_index_  = 0;  // Current playback position
    bool                        replay_paused_ = true;
    int                         replay_speed_ms_ = 500; // ms between auto-steps
    std::chrono::steady_clock::time_point replay_last_tick_;

    std::string selected_layer_name_; // To track the currently selected layer

    // ─── CLI state ─────────────────────────────────────────────
    std::string                 cli_initial_prompt_; // Auto-run prompt from -p
};

} // namespace neuralscope
