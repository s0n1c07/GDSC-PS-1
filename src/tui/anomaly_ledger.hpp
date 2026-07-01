#pragma once

#include "core/types.hpp"
#include "core/ring_buffer.hpp"
#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>

namespace neuralscope {

/// Panel 5: Scrollable log of detected numerical anomalies.
class AnomalyLedger {
public:
    AnomalyLedger(RingBuffer<AnomalyEntry>& buffer);

    /// Get FTXUI component.
    ftxui::Component component();

    /// Get FTXUI element for rendering.
    ftxui::Element render();

private:
    RingBuffer<AnomalyEntry>& buffer_;
    static constexpr int MAX_VISIBLE = 10;
};

} // namespace neuralscope
