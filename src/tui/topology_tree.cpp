#include "tui/topology_tree.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/dom/elements.hpp>

using namespace ftxui;

namespace neuralscope {

TopologyTree::TopologyTree() {}

void TopologyTree::set_topology(const TopologyNode& root) {
    root_ = root;
    flat_nodes_.clear();
    flatten_tree(root_, flat_nodes_);
    selected_index_ = 0;
    if (!flat_nodes_.empty()) {
        selected_layer_ = flat_nodes_[0].second->name;
    }
}

void TopologyTree::set_active_layer(const std::string& layer_name) {
    active_layer_ = layer_name;
}

std::string TopologyTree::get_selected_layer() const {
    return selected_layer_;
}

void TopologyTree::set_on_select(std::function<void(const std::string&)> cb) {
    on_select_ = std::move(cb);
}

void TopologyTree::flatten_tree(const TopologyNode& node,
    std::vector<std::pair<int, const TopologyNode*>>& flat)
{
    flat.push_back({node.depth, &node});
    if (node.expanded) {
        for (const auto& child : node.children) {
            flatten_tree(child, flat);
        }
    }
}

class TopologyTreeImpl : public ComponentBase {
public:
    TopologyTree* parent_;
    TopologyTreeImpl(TopologyTree* p) : parent_(p) {}

    Element Render() override {
        return parent_->render(Focused());
    }

    bool Focusable() const override { return true; }

    bool OnEvent(Event event) override {
        if (parent_->flat_nodes_.empty()) return false;

        if (event.is_mouse()) {
            if (event.mouse().button == Mouse::Left && event.mouse().motion == Mouse::Pressed) {
                TakeFocus();
                return true;
            }
            if (event.mouse().button == Mouse::WheelDown) {
                parent_->selected_index_ = std::min(parent_->selected_index_ + 1,
                    static_cast<int>(parent_->flat_nodes_.size()) - 1);
                parent_->selected_layer_ = parent_->flat_nodes_[parent_->selected_index_].second->name;
                if (parent_->on_select_) parent_->on_select_(parent_->selected_layer_);
                return true;
            }
            if (event.mouse().button == Mouse::WheelUp) {
                parent_->selected_index_ = std::max(parent_->selected_index_ - 1, 0);
                parent_->selected_layer_ = parent_->flat_nodes_[parent_->selected_index_].second->name;
                if (parent_->on_select_) parent_->on_select_(parent_->selected_layer_);
                return true;
            }
        }

        // j/k or arrow keys for navigation
        if (event == Event::Character('j') || event == Event::ArrowDown) {
            parent_->selected_index_ = std::min(parent_->selected_index_ + 1,
                static_cast<int>(parent_->flat_nodes_.size()) - 1);
            parent_->selected_layer_ = parent_->flat_nodes_[parent_->selected_index_].second->name;
            if (parent_->on_select_) parent_->on_select_(parent_->selected_layer_);
            return true;
        }
        if (event == Event::Character('k') || event == Event::ArrowUp) {
            parent_->selected_index_ = std::max(parent_->selected_index_ - 1, 0);
            parent_->selected_layer_ = parent_->flat_nodes_[parent_->selected_index_].second->name;
            if (parent_->on_select_) parent_->on_select_(parent_->selected_layer_);
            return true;
        }

        // Space: toggle expand/collapse
        if (event == Event::Character(' ')) {
            if (parent_->selected_index_ < static_cast<int>(parent_->flat_nodes_.size())) {
                auto* node = const_cast<TopologyNode*>(
                    parent_->flat_nodes_[parent_->selected_index_].second);
                if (!node->children.empty()) {
                    node->expanded = !node->expanded;
                    parent_->flat_nodes_.clear();
                    parent_->flatten_tree(parent_->root_, parent_->flat_nodes_);
                    parent_->selected_index_ = std::min(parent_->selected_index_,
                        static_cast<int>(parent_->flat_nodes_.size()) - 1);
                }
            }
            return true;
        }

        return false;
    }
};

Component TopologyTree::component() {
    return Make<TopologyTreeImpl>(this);
}

Element TopologyTree::render(bool is_focused) {
    std::vector<Element> lines;

    for (size_t i = 0; i < flat_nodes_.size(); ++i) {
        auto [depth, node] = flat_nodes_[i];
        bool is_selected = (static_cast<int>(i) == selected_index_);
        bool is_active   = (!active_layer_.empty() &&
                           node->name.find(active_layer_) != std::string::npos);

        // Build indentation
        std::string indent;
        for (int d = 0; d < depth; ++d) indent += "  ";

        // Expand/collapse indicator
        std::string prefix;
        if (!node->children.empty()) {
            prefix = node->expanded ? "\xe2\x96\xbc " : "\xe2\x96\xb6 ";  // ▼ / ▶
        } else {
            prefix = "\xe2\x97\x8f ";  // ●
        }

        // Color based on layer type
        auto type_color = [](LayerType t) -> Color {
            switch (t) {
                case LayerType::Attention: return Color::Cyan;
                case LayerType::MLP:       return Color::Magenta1;
                case LayerType::Norm:      return Color::Yellow;
                case LayerType::Embedding: return Color::Green;
                case LayerType::Output:    return Color::Red;
                default:                   return Color::GrayDark;
            }
        };

        std::string label = indent + prefix + node->type_label;
        if (is_active) {
            label += " [Active]";
        }

        auto line_elem = text(label);

        // Style
        if (is_selected) {
            line_elem = line_elem | inverted | bold;
        } else if (is_active) {
            line_elem = line_elem | bold |
                        color(Color::GreenLight);
        } else {
            line_elem = line_elem |
                        color(type_color(node->type));
        }

        lines.push_back(line_elem);
    }

    if (lines.empty()) {
        lines.push_back(text("  No model loaded") | dim);
    }

    return vbox(std::move(lines)) | vscroll_indicator | flex;
}

} // namespace neuralscope
