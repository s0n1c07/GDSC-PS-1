#include "tui/anomaly_ledger.hpp"
#include "core/utils.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>

#include <algorithm>

using namespace ftxui;

namespace neuralscope {

AnomalyLedger::AnomalyLedger(RingBuffer<AnomalyEntry>& buffer)
    : buffer_(buffer)
{}

Component AnomalyLedger::component() {
    return Renderer([this] { return render(); });
}

Element AnomalyLedger::render() {
    auto entries = buffer_.last_n(MAX_VISIBLE);

    std::vector<Element> lines;

    if (entries.empty()) {
        lines.push_back(
            text("  No anomalies detected") | dim | color(Color::Green));
        lines.push_back(
            text("  System nominal") | dim | color(Color::Green));
    } else {
        for (const auto& entry : entries) {
            Color entry_color;
            std::string icon;

            switch (entry.severity) {
                case Severity::Info:
                    entry_color = Color::Blue;
                    icon = "\xe2\x84\xb9 ";  // ℹ
                    break;
                case Severity::Warning:
                    entry_color = Color::Yellow;
                    icon = "\xe2\x9a\xa0 ";  // ⚠
                    break;
                case Severity::Critical:
                    entry_color = Color::Red;
                    icon = "\xe2\x9c\x96 ";  // ✖
                    break;
            }

            auto line = hbox({
                text(" " + entry.timestamp + " ") | dim,
                text(icon) | color(entry_color) | bold,
                text(entry.message) | color(entry_color),
            });

            lines.push_back(line);
        }
    }

    // Fill remaining space
    while (lines.size() < MAX_VISIBLE) {
        lines.push_back(text(""));
    }

    return vbox(std::move(lines)) | flex |
           vscroll_indicator | frame |
           borderStyled(ROUNDED) | color(Color::GrayDark);
}

} // namespace neuralscope
