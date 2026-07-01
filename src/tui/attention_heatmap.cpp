#include "tui/attention_heatmap.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/dom/elements.hpp>

#include <algorithm>
#include <cmath>
#include <string>

using namespace ftxui;

namespace neuralscope {

AttentionHeatmap::AttentionHeatmap() {}

void AttentionHeatmap::set_data(const std::vector<float>& weights, int size,
                                 int num_heads,
                                 const std::vector<std::string>& tokens)
{
    weights_      = weights;
    matrix_size_  = size;
    num_heads_    = std::max(1, num_heads);
    current_head_ = std::min(current_head_, num_heads_ - 1);
    token_labels_ = tokens;
    has_data_     = !weights.empty() && size > 0;

    // Reset viewport to origin
    viewport_x_ = 0;
    viewport_y_ = 0;
}

void AttentionHeatmap::clear() {
    weights_.clear();
    matrix_size_ = 0;
    has_data_    = false;
}

float AttentionHeatmap::get_weight(int row, int col) const {
    if (!has_data_ || matrix_size_ == 0) return 0.0f;

    // Index into the flattened attention matrix for the current head
    int head_offset = current_head_ * matrix_size_ * matrix_size_;
    int idx = head_offset + row * matrix_size_ + col;

    if (idx < 0 || idx >= static_cast<int>(weights_.size())) {
        return 0.0f;
    }
    return weights_[idx];
}

std::string AttentionHeatmap::weight_to_block(float weight) const {
    if (weight < contrast_low_)  return "\xe2\x96\x91\xe2\x96\x91"; // ░░
    if (weight < contrast_mid_)  return "\xe2\x96\x92\xe2\x96\x92"; // ▒▒
    if (weight < contrast_high_) return "\xe2\x96\x93\xe2\x96\x93"; // ▓▓
    return "\xe2\x96\x88\xe2\x96\x88"; // ██
}

Color AttentionHeatmap::weight_to_color(float weight) const {
    if (weight < contrast_low_)  return Color::GrayDark;
    if (weight < contrast_mid_)  return Color::Blue;
    if (weight < contrast_high_) return Color::Cyan;
    return Color::Yellow;
}

Component AttentionHeatmap::component() {
    return CatchEvent(Renderer([this] { return render(); }),
        [this](Event event) -> bool {
            // Arrow keys / hjkl for panning viewport
            if (event == Event::ArrowLeft || event == Event::Character('h')) {
                viewport_x_ = std::max(0, viewport_x_ - 1);
                return true;
            }
            if (event == Event::ArrowRight || event == Event::Character('l')) {
                viewport_x_ = std::min(
                    std::max(0, matrix_size_ - VIEWPORT_SIZE), viewport_x_ + 1);
                return true;
            }
            if (event == Event::ArrowUp || event == Event::Character('k')) {
                viewport_y_ = std::max(0, viewport_y_ - 1);
                return true;
            }
            if (event == Event::ArrowDown || event == Event::Character('j')) {
                viewport_y_ = std::min(
                    std::max(0, matrix_size_ - VIEWPORT_SIZE), viewport_y_ + 1);
                return true;
            }

            // [ / ] for head cycling
            if (event == Event::Character('[')) {
                current_head_ = std::max(0, current_head_ - 1);
                return true;
            }
            if (event == Event::Character(']')) {
                current_head_ = std::min(num_heads_ - 1, current_head_ + 1);
                return true;
            }

            // + / - for contrast adjustment
            if (event == Event::Character('+') || event == Event::Character('=')) {
                contrast_low_  = std::max(0.01f, contrast_low_ - 0.05f);
                contrast_mid_  = std::max(contrast_low_ + 0.05f, contrast_mid_ - 0.05f);
                contrast_high_ = std::max(contrast_mid_ + 0.05f, contrast_high_ - 0.05f);
                return true;
            }
            if (event == Event::Character('-')) {
                contrast_high_ = std::min(0.99f, contrast_high_ + 0.05f);
                contrast_mid_  = std::min(contrast_high_ - 0.05f, contrast_mid_ + 0.05f);
                contrast_low_  = std::min(contrast_mid_ - 0.05f, contrast_low_ + 0.05f);
                return true;
            }

            return false;
        });
}

Element AttentionHeatmap::render() {
    if (!has_data_) {
        return vbox({
            text("  No attention data available") | dim | center,
            text("  Select an attention layer from the topology") | dim | center,
        }) | flex | borderStyled(ROUNDED) | color(Color::GrayDark);
    }

    std::vector<Element> rows;

    // Title with head info
    auto title = hbox({
        text(" ATTENTION MATRIX ") | bold | color(Color::Cyan),
        text(" HEAD " + std::to_string(current_head_) + "/" +
             std::to_string(num_heads_ - 1)) | dim,
        text("  Viewport: [" + std::to_string(viewport_x_) + "-" +
             std::to_string(viewport_x_ + VIEWPORT_SIZE) + "] x [" +
             std::to_string(viewport_y_) + "-" +
             std::to_string(viewport_y_ + VIEWPORT_SIZE) + "]") | dim,
    });
    rows.push_back(title);

    // Column headers (token labels)
    {
        std::string header_str = "       "; // Row label space
        int end_col = std::min(viewport_x_ + VIEWPORT_SIZE, matrix_size_);
        for (int c = viewport_x_; c < end_col; ++c) {
            if (c < static_cast<int>(token_labels_.size())) {
                std::string label = token_labels_[c];
                if (label.size() > 2) label = label.substr(0, 2);
                header_str += " " + label;
            } else {
                header_str += "  " + std::to_string(c % 100);
            }
        }
        rows.push_back(text(header_str) | dim);
    }

    // Matrix rows
    int end_row = std::min(viewport_y_ + VIEWPORT_SIZE, matrix_size_);
    int end_col = std::min(viewport_x_ + VIEWPORT_SIZE, matrix_size_);

    for (int r = viewport_y_; r < end_row; ++r) {
        std::vector<Element> cells;

        // Row label
        std::string row_label;
        if (r < static_cast<int>(token_labels_.size())) {
            row_label = token_labels_[r];
            if (row_label.size() > 5) row_label = row_label.substr(0, 5);
            while (row_label.size() < 6) row_label += " ";
        } else {
            row_label = std::to_string(r);
            while (row_label.size() < 6) row_label += " ";
        }
        cells.push_back(text(row_label) | dim | size(WIDTH, EQUAL, 7));

        // Weight cells
        for (int c = viewport_x_; c < end_col; ++c) {
            float w = get_weight(r, c);
            std::string block = weight_to_block(w);
            Color col = weight_to_color(w);
            cells.push_back(text(block) | color(col));
        }

        rows.push_back(hbox(std::move(cells)));
    }

    // Controls hint
    rows.push_back(separator());
    rows.push_back(hbox({
        text(" [h/j/k/l]: Pan ") | dim,
        text(" [+/-]: Contrast ") | dim,
        text(" [ / ]: Head ") | dim,
    }));

    return vbox(std::move(rows)) | flex |
           borderStyled(ROUNDED) | color(Color::GrayDark);
}

} // namespace neuralscope
