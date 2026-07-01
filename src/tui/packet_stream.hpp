#pragma once

#include "core/types.hpp"
#include "core/ring_buffer.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>

#include <atomic>
#include <string>
#include <vector>

namespace neuralscope {

/// Panel 2: Live scrolling table of tensor evaluation events.
class PacketStream {
public:
    PacketStream(RingBuffer<LayerSnapshot>& buffer);

    /// Get the FTXUI component.
    ftxui::Component component();

    /// Get the FTXUI element for rendering.
    ftxui::Element render();

    /// Toggle pause/resume auto-scroll.
    void toggle_pause();
    bool is_paused() const;

private:
    RingBuffer<LayerSnapshot>& buffer_;
    std::atomic<bool>          paused_{false};
    int                        scroll_offset_ = 0;
    static constexpr int       MAX_VISIBLE_ROWS = 15;
};

} // namespace neuralscope
