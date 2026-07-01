#pragma once

#include "core/types.hpp"
#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>
#include <string>
#include <vector>
#include <functional>

namespace neuralscope {

/// Panel 1: Model topology tree with vim-like navigation.
class TopologyTree {
public:
    TopologyTree();

    /// Set the root topology node.
    void set_topology(const TopologyNode& root);

    /// Set the currently active layer (from hook manager).
    void set_active_layer(const std::string& layer_name);

    /// Get the selected layer name.
    std::string get_selected_layer() const;

    /// Set callback when selection changes.
    void set_on_select(std::function<void(const std::string&)> cb);

    /// Get the FTXUI component.
    ftxui::Component component();

    /// Get the FTXUI element for rendering.
    ftxui::Element render(bool is_focused = false);

private:
    friend class TopologyTreeImpl;

    /// Flatten the tree into a displayable list.
    void flatten_tree(const TopologyNode& node,
                      std::vector<std::pair<int, const TopologyNode*>>& flat);

    TopologyNode root_;
    int          selected_index_ = 0;
    std::string  active_layer_;
    std::string  selected_layer_;

    std::vector<std::pair<int, const TopologyNode*>> flat_nodes_;
    std::function<void(const std::string&)> on_select_;
};

} // namespace neuralscope
