#pragma once

#include "core/types.hpp"
#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>
#include <vector>
#include <string>

namespace neuralscope {

/// Panel 3: Attention matrix visualizer with Unicode block characters.
/// Fixed 16x16 viewport with pan/scroll support.
class AttentionHeatmap {
public:
    AttentionHeatmap();

    /// Set the attention weights and metadata.
    void set_data(const std::vector<float>& weights, int size,
                  int num_heads, const std::vector<std::string>& tokens = {});

    /// Clear the visualization.
    void clear();

    /// Get the FTXUI component.
    ftxui::Component component();

    /// Get the FTXUI element for rendering.
    ftxui::Element render();

private:
    std::vector<float>       weights_;
    int                      matrix_size_ = 0;    // NxN
    int                      num_heads_ = 1;
    int                      current_head_ = 0;
    std::vector<std::string> token_labels_;

    // Viewport (16x16 window into the matrix)
    int viewport_x_ = 0;
    int viewport_y_ = 0;
    static constexpr int VIEWPORT_SIZE = 16;

    // Contrast thresholds
    float contrast_low_  = 0.1f;
    float contrast_mid_  = 0.3f;
    float contrast_high_ = 0.6f;

    bool has_data_ = false;

    /// Get the attention weight at (row, col) for current head.
    float get_weight(int row, int col) const;

    /// Map weight to Unicode block character.
    std::string weight_to_block(float weight) const;

    /// Map weight to color.
    ftxui::Color weight_to_color(float weight) const;
};

} // namespace neuralscope
