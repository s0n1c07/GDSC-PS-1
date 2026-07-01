#include "tui/packet_stream.hpp"
#include "core/utils.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/dom/table.hpp>

#include <algorithm>
#include <string>

using namespace ftxui;

namespace neuralscope {

PacketStream::PacketStream(RingBuffer<LayerSnapshot>& buffer)
    : buffer_(buffer)
{}

Component PacketStream::component() {
    return CatchEvent(Renderer([this] { return render(); }),
        [this](Event event) -> bool {
            if (event == Event::Character('p') ||
                event == Event::Character('P')) {
                toggle_pause();
                return true;
            }

            // Scroll when paused
            if (paused_.load()) {
                if (event == Event::Character('j') ||
                    event == Event::ArrowDown) {
                    scroll_offset_ = std::max(0, scroll_offset_ - 1);
                    return true;
                }
                if (event == Event::Character('k') ||
                    event == Event::ArrowUp) {
                    scroll_offset_++;
                    return true;
                }
            }
            return false;
        });
}

Element PacketStream::render() {
    auto snapshots = buffer_.last_n(MAX_VISIBLE_ROWS + scroll_offset_);

    // Skip scroll_offset items from the front (most recent)
    size_t start = 0;
    if (paused_.load() && scroll_offset_ > 0) {
        start = std::min(static_cast<size_t>(scroll_offset_), snapshots.size());
    }

    // Header
    std::vector<Element> rows;

    // Title bar
    auto header = hbox({
        text(" ID ") | bold | size(WIDTH, EQUAL, 7),
        separator(),
        text(" TIMESTAMP ") | bold | size(WIDTH, EQUAL, 14),
        separator(),
        text(" LAYER TYPE ") | bold | size(WIDTH, EQUAL, 14),
        separator(),
        text(" LATENCY ") | bold | size(WIDTH, EQUAL, 12),
        separator(),
        text(" DEVICE ") | bold | size(WIDTH, EQUAL, 8),
    }) | color(Color::Cyan);

    rows.push_back(header);
    rows.push_back(separator());

    // Data rows (most recent first, then reversed to show oldest-to-newest)
    size_t end = std::min(snapshots.size(), start + MAX_VISIBLE_ROWS);
    for (size_t i = start; i < end; ++i) {
        const auto& snap = snapshots[i];

        // Latency color
        Color lat_color;
        int level = latency_color_level(snap.latency_ms);
        if (level == 0) lat_color = Color::Green;
        else if (level == 1) lat_color = Color::Yellow;
        else lat_color = Color::Red;

        // Layer type color
        Color type_color;
        switch (snap.layer_type) {
            case LayerType::Attention: type_color = Color::Cyan; break;
            case LayerType::MLP:       type_color = Color::Magenta1; break;
            case LayerType::Norm:      type_color = Color::Yellow; break;
            case LayerType::Embedding: type_color = Color::Green; break;
            case LayerType::Output:    type_color = Color::Red; break;
            default:                   type_color = Color::GrayDark; break;
        }

        auto row = hbox({
            text(" " + std::to_string(snap.id) + " ") |
                size(WIDTH, EQUAL, 7),
            separator(),
            text(" " + snap.timestamp + " ") |
                size(WIDTH, EQUAL, 14) | dim,
            separator(),
            text(" " + std::string(layer_type_str(snap.layer_type)) + " ") |
                size(WIDTH, EQUAL, 14) | color(type_color),
            separator(),
            text(" " + format_float(static_cast<float>(snap.latency_ms), 2) + "ms ") |
                size(WIDTH, EQUAL, 12) | color(lat_color),
            separator(),
            text(" " + snap.compute_device + " ") |
                size(WIDTH, EQUAL, 8),
        });

        rows.push_back(row);
    }

    // Fill remaining rows if not enough data
    while (rows.size() < MAX_VISIBLE_ROWS + 2) {
        rows.push_back(text("") | size(HEIGHT, EQUAL, 1));
    }

    // Pause indicator
    std::string status = paused_.load()
        ? " [PAUSED - j/k to scroll] "
        : " [P to pause] ";

    auto status_elem = text(status) |
        (paused_.load() ? color(Color::Yellow) : dim);

    return vbox({
        vbox(std::move(rows)),
        separator(),
        status_elem,
    }) | flex | borderStyled(ROUNDED) | color(Color::GrayDark);
}

void PacketStream::toggle_pause() {
    bool expected = paused_.load();
    paused_.store(!expected);
    if (!paused_.load()) {
        scroll_offset_ = 0; // Reset scroll on resume
    }
}

bool PacketStream::is_paused() const {
    return paused_.load();
}

} // namespace neuralscope
